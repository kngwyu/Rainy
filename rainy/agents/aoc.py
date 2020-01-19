"""
This module has an implementation of AOC, an A2C-like variant of option-critic.
Corresponding papers:
- The Option-Critic Architecture
  - https://arxiv.org/abs/1609.05140
- When Waiting is not an Option : Learning Options with a Deliberation Cost
  - https://arxiv.org/abs/1709.04571
"""

import numpy as np
import torch
from torch import BoolTensor, LongTensor, Tensor
from typing import List, Optional, Tuple
from .base import A2CLikeAgent, Netout
from ..config import Config
from ..lib.explore import EpsGreedy
from ..lib.rollout import RolloutStorage
from ..net import OptionCriticNet
from ..net.policy import BernoulliPolicy, CategoricalPolicy, Policy
from ..prelude import Action, Array, State
from ..utils import Device


class AOCRolloutStorage(RolloutStorage[State]):
    def __init__(
        self, nsteps: int, nworkers: int, device: Device, num_options: int,
    ) -> None:
        super().__init__(nsteps, nworkers, device)
        self.options = [self.device.zeros(self.nworkers, dtype=torch.long)]
        self.is_new_options = [self.device.ones(self.nworkers, dtype=torch.bool)]
        self.epsilons: List[float] = []
        self.option_mus: List[CategoricalPolicy] = []
        self.beta_adv = torch.zeros_like(self.batch_values)
        self.noptions = num_options
        self.worker_indices = self.device.indices(self.nworkers)
        self._beta_adv = self._beta_adv_eps

    def _use_mu_for_beta_adv(self) -> None:
        self._beta_adv = self._beta_adv_mu

    def reset(self) -> None:
        super().reset()
        self.options = [self.options[-1]]
        self.is_new_options = [self.is_new_options[-1]]
        self.epsilons.clear()
        self.option_mus.clear()

    def push(
        self,
        *args,
        options: LongTensor,
        is_new_options: Tensor,
        epsilon: Optional[float] = None,
        mu: Optional[CategoricalPolicy] = None,
        **kwargs,
    ) -> None:
        super().push(*args, **kwargs)
        self.options.append(options)
        self.is_new_options.append(is_new_options)
        if epsilon is None:
            self.epsilons.append(1.0)
        else:
            self.epsilons.append(epsilon)
        if mu is not None:
            self.option_mus.append(mu)

    def batch_options(self) -> Tuple[Tensor, Tensor]:
        batched = torch.cat(self.options, dim=0)
        return batched[: -self.nworkers], batched[self.nworkers :]

    def _beta_adv_eps(self, i: int, opt_q: Tensor, options: LongTensor) -> Tensor:
        eps = self.epsilons[i]
        v = (1 - eps) * opt_q.max(dim=-1)[0] + eps * opt_q.mean(dim=-1)
        return opt_q[self.worker_indices, options] - v

    def _beta_adv_mu(self, i: int, opt_q: Tensor, options: LongTensor) -> Tensor:
        probs = self.option_mus[i].dist.probs
        v = (opt_q * probs).sum(dim=-1)
        return opt_q[self.worker_indices, options] - v

    def calc_ac_returns(
        self, next_value: Tensor, gamma: float, delib_cost: float
    ) -> None:
        self.returns[-1] = next_value
        rewards = self.device.tensor(self.rewards)
        for i in reversed(range(self.nsteps)):
            ret = gamma * self.masks[i + 1] * self.returns[i + 1] + rewards[i]
            self.returns[i] = (
                ret - self.is_new_options[i].float() * self.masks[i] * delib_cost
            )
            opt_q, opt = self.values[i], self.options[i + 1]
            self.advs[i] = self.returns[i] - opt_q[self.worker_indices, opt]
            self.beta_adv[i] = self._beta_adv(i, opt_q, opt)

    def calc_gae_returns(
        self, next_v: Tensor, gamma: float, lambda_: float, delib_cost: float,
    ) -> None:
        self.returns[-1] = next_v
        rewards = self.device.tensor(self.rewards)
        self.advs.fill_(0.0)
        value_i1 = next_v
        for i in reversed(range(self.nsteps)):
            opt, opt_q = self.options[i + 1], self.values[i]
            value_i = opt_q[self.worker_indices, opt]

            # GAE
            gamma_i1 = gamma * self.masks[i + 1]
            td_error = rewards[i] + gamma_i1 * value_i1 - value_i
            gamma_lambda_i = gamma * lambda_ * self.masks[i]
            delib_cost_i = delib_cost * self.masks[i] * self.is_new_options[i].float()
            self.advs[i] = td_error + gamma_lambda_i * self.advs[i + 1] - delib_cost_i
            self.returns[i] = self.advs[i] + value_i
            value_i1 = value_i

            # β-advantage
            self.beta_adv[i] = self._beta_adv(i, opt_q, opt)


class AOCAgent(A2CLikeAgent[State]):
    """AOC: Adavantage Option Critic
    It's a synchronous batched version of A2OC: Asynchronou Adavantage Option Critic
    """

    EPS = 0.001
    SAVED_MEMBERS = "net", "opt_explorer"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.net: OptionCriticNet = config.net("option-critic")  # type: ignore
        self.noptions = self.net.num_options
        self.optimizer = config.optimizer(self.net.parameters())
        self.worker_indices = config.device.indices(config.nworkers)
        self.batch_indices = config.device.indices(config.batch_size)
        self.storage: AOCRolloutStorage[State] = AOCRolloutStorage(
            config.nsteps, config.nworkers, config.device, self.noptions
        )
        self.opt_explorer: EpsGreedy = config.explorer()
        self.eval_opt_explorer: EpsGreedy = config.explorer(key="eval")
        if not isinstance(self.opt_explorer, EpsGreedy) or not isinstance(
            self.eval_opt_explorer, EpsGreedy
        ):
            return ValueError("Currently only Epsilon Greedy is supported as Explorer")
        self.eval_prev_options: LongTensor = config.device.zeros(
            config.nworkers, dtype=torch.long
        )

    def _reset(self, initial_states: Array[State]) -> None:
        self.storage.set_initial_state(initial_states)

    def eval_reset(self) -> None:
        self.eval_prev_options.fill_(0)

    def _sample_options(
        self,
        opt_q: Tensor,
        beta: BernoulliPolicy,
        prev_options: LongTensor,
        evaluate: bool = False,
    ) -> Tuple[LongTensor, BoolTensor]:
        if evaluate:
            explorer = self.eval_opt_explorer
        else:
            explorer = self.opt_explorer
        current_beta = beta[self.worker_indices, prev_options]
        do_options_end = current_beta.action().bool()
        is_initial_states = (1.0 - self.storage.masks[-1]).bool()
        use_new_options = do_options_end | is_initial_states
        epsgreedy_options = explorer.select_from_value(opt_q, same_device=True)
        options = torch.where(use_new_options, epsgreedy_options, prev_options)
        return options, use_new_options  # type: ignore

    @torch.no_grad()
    def _eval_policy(self, states: Array) -> Policy:
        opt_policy, opt_q, beta = self.net(states)
        options, _ = self._sample_options(
            opt_q, beta, self.eval_prev_options, evaluate=True
        )
        self.eval_prev_options = options
        return opt_policy[self.worker_indices, options]

    def eval_action(self, state: Array, net_outputs: Optional[Netout] = None) -> Action:
        if len(state.shape) == len(self.net.state_dim):
            # treat as batch_size == nworkers
            state = np.stack([state] * self.config.nworkers)
        policy = self._eval_policy(state)
        return policy[0].eval_action(self.config.eval_deterministic)

    def eval_action_parallel(self, states: Array, mask: torch.Tensor) -> Array[Action]:
        policy = self._eval_policy(states)
        return policy.eval_action(self.config.eval_deterministic)

    @property
    def prev_options(self) -> LongTensor:
        return self.storage.options[-1]  # type: ignore

    @property
    def prev_is_new_options(self) -> BoolTensor:
        return self.storage.is_new_options[-1]  # type: ignore

    @torch.no_grad()
    def actions(self, states: Array[State]) -> Tuple[Array[Action], dict]:
        opt_policy, opt_q, beta = self.net(self.penv.extract(states))
        options, is_new_options = self._sample_options(opt_q, beta, self.prev_options)
        policy = opt_policy[self.worker_indices, options]
        actions = policy.action().squeeze().cpu().numpy()
        net_outputs = dict(
            policy=policy,
            value=opt_q,
            options=options,
            is_new_options=is_new_options,
            epsilon=1.0 if self.config.opt_avg_baseline else self.opt_explorer.epsilon,
        )
        return actions, net_outputs

    @torch.no_grad()
    def _next_value(self, states: Array[State]) -> Tensor:
        opt_q = self.net.opt_q(self.penv.extract(states))
        current_opt_q = opt_q[self.worker_indices, self.prev_options]
        eps = self.opt_explorer.epsilon
        next_opt_q = (1 - eps) * opt_q.max(dim=-1)[0] + eps * opt_q.mean(-1)
        return torch.where(self.prev_is_new_options, next_opt_q, current_opt_q)

    def train(self, last_states: Array[State]) -> None:
        next_v = self._next_value(last_states)
        if self.config.use_gae:
            self.storage.calc_gae_returns(
                next_v,
                self.config.discount_factor,
                self.config.gae_lambda,
                self.config.opt_delib_cost,
            )
        else:
            self.storage.calc_ac_returns(
                next_v, self.config.discount_factor, self.config.opt_delib_cost
            )

        prev_options, options = self.storage.batch_options()
        adv = self.storage.advs[:-1].flatten()
        beta_adv = self.storage.beta_adv.flatten().add_(
            self.config.opt_delib_cost + self.config.opt_beta_adv_merginal
        )
        ret = self.storage.returns[:-1].flatten()
        masks = self.storage.batch_masks()

        opt_policy, opt_q, beta = self.net(self.storage.batch_states(self.penv))
        policy = opt_policy[self.batch_indices, prev_options]
        policy.set_action(self.storage.batch_actions())

        policy_loss = -(policy.log_prob() * adv).mean()
        term_prob = beta[self.batch_indices, prev_options].dist.probs
        beta_loss = term_prob.mul(masks).mul(beta_adv).mean()
        value = opt_q[self.batch_indices, options]
        value_loss = (value - ret).pow(2).mean()
        entropy = policy.entropy().mean()
        loss = (
            policy_loss
            + beta_loss
            + self.config.value_loss_weight * 0.5 * value_loss
            - self.config.entropy_weight * entropy
        )
        self._backward(loss, self.optimizer, self.net.parameters())
        self.network_log(
            policy_loss=policy_loss.item(),
            value=value.detach_().mean().item(),
            value_loss=value_loss.item(),
            beta=beta.dist.probs.detach_().mean().item(),
            beta_loss=beta_loss.item(),
            entropy=entropy.item(),
        )
        self.storage.reset()
