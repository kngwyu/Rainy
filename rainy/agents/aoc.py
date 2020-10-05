"""
This module has an implementation of AOC, an A2C-like variant of option-critic.
Corresponding papers:
- The Option-Critic Architecture
  - https://arxiv.org/abs/1609.05140
- When Waiting is not an Option : Learning Options with a Deliberation Cost
  - https://arxiv.org/abs/1709.04571
"""

from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import BoolTensor, LongTensor, Tensor

from ..config import Config
from ..lib.explore import EpsGreedy
from ..lib.rollout import RolloutStorage
from ..net import OptionCriticNet
from ..net.policy import BernoulliPolicy, CategoricalPolicy, Policy
from ..prelude import Action, Array, State
from ..utils import Device
from .base import A2CLikeAgent, Netout


class AOCRolloutStorage(RolloutStorage[State]):
    def __init__(
        self, nsteps: int, nworkers: int, device: Device, num_options: int,
    ) -> None:
        super().__init__(nsteps, nworkers, device)
        self.options = [self.device.zeros(self.nworkers, dtype=torch.long)]
        self.opt_terminals = [self.device.ones(self.nworkers, dtype=torch.bool)]
        self.epsilons: List[float] = []
        self.muos: List[CategoricalPolicy] = []
        self.beta_adv = torch.zeros_like(self.batch_values)
        self.noptions = num_options
        self.worker_indices = self.device.indices(self.nworkers)
        self._beta_adv = self._beta_adv_eps

    def _use_mu_for_beta_adv(self) -> None:
        self._beta_adv = self._beta_adv_mu

    def reset(self) -> None:
        super().reset()
        self.options = [self.options[-1]]
        self.opt_terminals = [self.opt_terminals[-1]]
        self.epsilons.clear()
        self.muos.clear()

    def initialize(self) -> None:
        super().reset()
        self.options = [self.device.zeros(self.nworkers, dtype=torch.long)]
        self.opt_terminals = [self.device.ones(self.nworkers, dtype=torch.bool)]
        self.epsilons.clear()
        self.muos.clear()

    def push(
        self,
        *args,
        options: LongTensor,
        opt_terminals: Tensor,
        epsilon: float = 1.0,
        mu: Optional[CategoricalPolicy] = None,
        **kwargs,
    ) -> None:
        super().push(*args, **kwargs)
        self.options.append(options)
        self.opt_terminals.append(opt_terminals)
        self.epsilons.append(epsilon)
        if mu is not None:
            self.muos.append(mu)

    def batch_options(self) -> Tuple[Tensor, Tensor]:
        batched = torch.cat(self.options, dim=0)
        return batched[: -self.nworkers], batched[self.nworkers :]

    def _beta_adv_eps(self, i: int, qo: Tensor, options: LongTensor) -> Tensor:
        eps = self.epsilons[i]
        vo = (1 - eps) * qo.max(dim=-1)[0] + eps * qo.mean(dim=-1)
        return qo[self.worker_indices, options] - vo

    def _beta_adv_mu(self, i: int, qo: Tensor, options: LongTensor) -> Tensor:
        probs = self.muos[i].dist.probs
        vo = torch.einsum("bo,bo->b", qo, probs)
        return qo[self.worker_indices, options] - vo

    def calc_ac_returns(self, next_uo: Tensor, gamma: float, delib_cost: float) -> None:
        self.returns[-1] = next_uo
        rewards = self.device.tensor(self.rewards)
        opt_terminals = self.device.zeros((self.nworkers,), dtype=torch.bool)
        for i in reversed(range(self.nsteps)):
            qo, opt = self.values[i], self.options[i + 1]
            eps = self.epsilons[i]
            # CAUTION: this can be only applied to ε-Greedy option selection
            vo = (1 - eps) * qo.max(dim=-1)[0] + eps * qo.mean(-1)
            ret_i1 = torch.where(opt_terminals, vo, self.returns[i + 1])
            self.returns[i] = gamma * self.masks[i + 1] * ret_i1 + rewards[i]
            self.advs[i] = self.returns[i] - qo[self.worker_indices, opt]
            delib_cost_i = self.opt_terminals[i + 1] * delib_cost
            self.beta_adv[i] = self._beta_adv(i, qo, opt) + delib_cost_i
            opt_terminals = self.opt_terminals[i + 1]

    def calc_gae_returns(
        self, next_uo: Tensor, gamma: float, lambda_: float, delib_cost: float, truncate: bool = False,
    ) -> None:
        self.returns[-1] = next_uo
        rewards = self.device.tensor(self.rewards)
        self.advs.fill_(0.0)
        qo_i1 = next_uo
        opt_terminals = self.device.zeros((self.nworkers), dtype=torch.bool)
        adv_zeros = self.device.zeros((self.nworkers))
        for i in reversed(range(self.nsteps)):
            opt, qo = self.options[i + 1], self.values[i]
            qo_i = qo[self.worker_indices, opt]
            # GAE
            gamma_i1 = gamma * self.masks[i + 1]
            td_error = rewards[i] + gamma_i1 * qo_i1 - qo_i
            gamma_lambda_i = gamma * lambda_ * self.masks[i]
            if truncate:
                adv_i1 = torch.where(opt_terminals, adv_zeros, self.advs[i + 1])
            else:
                adv_i1 = self.advs[i + 1]
            self.advs[i] = td_error + gamma_lambda_i * adv_i1
            self.returns[i] = self.advs[i] + qo_i
            qo_i1 = qo_i
            # β-advantage
            delib_cost_i = self.opt_terminals[i + 1] * delib_cost
            self.beta_adv[i] = self._beta_adv(i, qo, opt) + delib_cost_i
            opt_terminals = self.opt_terminals[i + 1]


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
            config.nworkers, dtype=torch.long,
        )

    def _reset(self, initial_states: Array[State]) -> None:
        self.storage.set_initial_state(initial_states)

    def eval_reset(self) -> None:
        self.eval_prev_options.fill_(0)

    def _sample_options(
        self,
        qo: Tensor,
        beta: BernoulliPolicy,
        prev_options: LongTensor,
        evaluation_phase: bool = False,
    ) -> Tuple[LongTensor, BoolTensor]:
        batch_size = qo.size(0)
        explorer = self.eval_opt_explorer if evaluation_phase else self.opt_explorer
        epsgreedy_options = explorer.select_from_value(qo, same_device=True)
        current_beta = beta[self.worker_indices[:batch_size], prev_options]
        do_options_end = current_beta.action().bool()
        if evaluation_phase:
            use_new_options = do_options_end
        else:
            is_initial_states = (1.0 - self.storage.masks[-1]).bool()
            use_new_options = do_options_end | is_initial_states
        options = torch.where(use_new_options, epsgreedy_options, prev_options)
        return options, do_options_end  # type: ignore

    @torch.no_grad()
    def _eval_policy(self, states: Array) -> Tuple[Policy, Tensor]:
        batch_size = states.shape[0]
        pio, qo, beta = self.net(states)
        options, _ = self._sample_options(
            qo, beta, self.eval_prev_options[:batch_size], evaluation_phase=True,
        )
        self.eval_prev_options[:batch_size] = options
        return pio, options

    def eval_action(self, state: Array, net_outputs: Optional[Netout] = None) -> Action:
        if state.ndim == len(self.net.state_dim):
            # treat as batch_size == nworkers
            state = np.stack([state] * self.config.nworkers)
        pio, options = self._eval_policy(state)
        if net_outputs is not None:
            net_outputs["options"] = self.eval_prev_options
        cond_pio = pio[0, options[0]]
        return cond_pio.eval_action(self.config.eval_deterministic)

    def eval_action_parallel(self, states: Array) -> Array[Action]:
        batch_size = states.shape[0]
        pio, options = self._eval_policy(states)
        cond_pio = pio[self.config.device.indices(batch_size), options]
        return cond_pio.eval_action(self.config.eval_deterministic)

    @property
    def prev_options(self) -> LongTensor:
        return self.storage.options[-1]  # type: ignore

    @torch.no_grad()
    def actions(self, states: Array[State]) -> Tuple[Array[Action], dict]:
        pio, qo, beta = self.net(self.penv.extract(states))
        options, opt_terminals = self._sample_options(qo, beta, self.prev_options)
        pi = pio[self.worker_indices, options]
        actions = pi.action().squeeze().cpu().numpy()
        net_outputs = dict(
            policy=pi,
            value=qo,
            options=options,
            opt_terminals=opt_terminals,
            epsilon=1.0 if self.config.opt_avg_baseline else self.opt_explorer.epsilon,
        )
        return actions, net_outputs

    @torch.no_grad()
    def _next_uo(self, states: Array[State]) -> Tensor:
        qo, beta = self.net.qo_and_beta(self.penv.extract(states))
        beta = beta.dist.probs[self.worker_indices, self.prev_options]
        qo_current = qo[self.worker_indices, self.prev_options]
        eps = self.opt_explorer.epsilon
        vo = (1 - eps) * qo.max(dim=-1)[0] + eps * qo.mean(-1)
        # Uo(s, o) = (1.0 - β) Qo(s, o) + β Vo(s)
        return (1.0 - beta) * qo_current + beta * vo

    def train(self, last_states: Array[State]) -> None:
        next_uo = self._next_uo(last_states)
        if self.config.use_gae:
            self.storage.calc_gae_returns(
                next_uo,
                self.config.discount_factor,
                self.config.gae_lambda,
                self.config.opt_delib_cost,
                self.config.truncate_advantage,
            )
        else:
            self.storage.calc_ac_returns(
                next_uo, self.config.discount_factor, self.config.opt_delib_cost,
            )

        prev_options, options = self.storage.batch_options()
        adv = self.storage.advs[:-1].flatten()
        beta_adv = self.storage.beta_adv.flatten().add_(
            self.config.opt_delib_cost + self.config.opt_beta_adv_merginal
        )
        ret = self.storage.returns[:-1].flatten()
        masks = self.storage.batch_masks()
        pio, qo, beta = self.net(self.storage.batch_states(self.penv))
        # β loss
        term_prob = beta[self.batch_indices, prev_options].dist.probs
        beta_loss = term_prob.mul(masks).mul(beta_adv).mean()
        # π loss
        pi = pio[self.batch_indices, prev_options]
        policy_loss = -(pi.log_prob(self.storage.batch_actions()) * adv).mean()
        # V loss
        qo_ = qo[self.batch_indices, options]
        value_loss = (qo_ - ret).pow(2).mean()
        # H(π) bonus
        entropy = pi.entropy().mean()
        loss = (
            policy_loss
            + beta_loss
            + self.config.value_loss_weight * 0.5 * value_loss
            - self.config.entropy_weight * entropy
        )
        self._backward(loss, self.optimizer, self.net.parameters())
        self.network_log(
            policy_loss=policy_loss.item(),
            value=qo_.detach_().mean().item(),
            value_loss=value_loss.item(),
            beta=beta.dist.probs.detach_().mean().item(),
            beta_loss=beta_loss.item(),
            entropy=entropy.item(),
        )
        self.storage.reset()
