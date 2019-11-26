import numpy as np
import torch
from torch import ByteTensor, LongTensor, nn, Tensor
from typing import List, Optional, Tuple
from .base import NStepParallelAgent
from ..config import Config
from ..lib.explore import EpsGreedy
from ..lib.rollout import RolloutStorage
from ..net import OptionCriticNet
from ..net.policy import Policy, BernoulliPolicy
from ..prelude import Action, Array, State
from ..utils import Device


class AOCRolloutStorage(RolloutStorage[State]):
    def __init__(
        self, nsteps: int, nworkers: int, device: Device, num_options: int
    ) -> None:
        super().__init__(nsteps, nworkers, device)
        self.options = [self.device.zeros(self.nworkers, dtype=torch.long)]
        self.is_new_options = [self.device.ones(self.nworkers, dtype=torch.uint8)]
        self.epsilons: List[float] = []
        self.beta_adv = torch.zeros_like(self.batch_values)
        self.noptions = num_options
        self.worker_indices = self.device.indices(self.nworkers)

    def reset(self) -> None:
        super().reset()
        self.options = [self.options[-1]]
        self.is_new_options = [self.is_new_options[-1]]
        self.epsilons.clear()

    def push_options(
        self, option: LongTensor, is_new_option: Tensor, epsilon: float
    ) -> None:
        self.options.append(option)
        self.is_new_options.append(is_new_option)
        self.epsilons.append(epsilon)

    def batch_options(self) -> Tuple[Tensor, Tensor]:
        batched = torch.cat(self.options, dim=0)
        return batched[: -self.nworkers], batched[self.nworkers :]

    def calc_returns(self, next_value: Tensor, gamma: float, delib_cost: float) -> None:
        self.returns[-1] = next_value
        rewards = self.device.tensor(self.rewards)
        for i in reversed(range(self.nsteps)):
            ret = gamma * self.masks[i + 1] * self.returns[i + 1] + rewards[i]
            self.returns[i] = (
                ret - self.is_new_options[i].float() * self.masks[i] * delib_cost
            )
            opt = self.options[i + 1]
            opt_q, eps = self.values[i], self.epsilons[i]
            self.advs[i] = self.returns[i] - opt_q[self.worker_indices, opt]
            v = (1 - eps) * opt_q.max(dim=-1)[0] + eps * opt_q.mean(dim=-1)
            self.beta_adv[i] = opt_q[self.worker_indices, opt].mul(v)


class AOCAgent(NStepParallelAgent[State]):
    """AOC: Adavantage Option Critic
    It's a synchronous batched version of A2OC: Asynchronou Adavantage Option Critic
    """

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
        self.opt_explorer: EpsGreedy = config.explorer()  # type: ignore
        if not isinstance(self.opt_explorer, EpsGreedy):
            return ValueError("Currently only Epsilon Greedy is supported as Explorer")
        self.eval_prev_options: LongTensor = config.device.zeros(
            config.nworkers, dtype=torch.long
        )

    def _reset(self, initial_states: Array[State]) -> None:
        self.storage.set_initial_state(initial_states)

    def eval_reset(self) -> None:
        self.eval_prev_options.fill_(0)

    def sample_options(
        self, opt_q: Tensor, beta: BernoulliPolicy, prev_options: LongTensor,
    ) -> Tuple[LongTensor, ByteTensor]:
        current_beta = beta[self.worker_indices, prev_options]
        do_options_end = current_beta.action()
        use_new_options = do_options_end.add(1.0 - self.storage.masks[-1]) > 0.001
        epsgreedy_options = self.opt_explorer.select_from_value(opt_q).to(
            prev_options.device
        )
        options = torch.where(use_new_options, epsgreedy_options, prev_options)
        return options, use_new_options  # type: ignore

    @torch.no_grad()
    def _eval_policy(self, states: Array, indices: Tensor) -> Policy:
        opt_policy, opt_q, beta = self.net(states)
        options, _ = self.sample_options(opt_q, beta, self.eval_prev_options)
        self.eval_prev_options = options
        return opt_policy[indices, options]

    def eval_action(self, state: Array) -> Action:
        if len(state.shape) == len(self.net.state_dim):
            # treat as batch_size == nworkers
            state = np.stack([state] * self.config.nworkers)
        policy = self._eval_policy(
            state, self.config.device.tensor([0], dtype=torch.long)
        )
        return policy[0].eval_action(self.config.eval_deterministic)

    def eval_action_parallel(
        self, states: Array, mask: torch.Tensor, ent: Optional[Array[float]] = None
    ) -> Array[Action]:
        policy = self._eval_policy(states, self.worker_indices)
        if ent is not None:
            ent += policy.entropy().cpu().numpy()
        return policy.eval_action(self.config.eval_deterministic)

    @property
    def prev_options(self) -> LongTensor:
        return self.storage.options[-1]  # type: ignore

    @property
    def prev_is_new_options(self) -> ByteTensor:
        return self.storage.is_new_options[-1]  # type: ignore

    @torch.no_grad()
    def _one_step(self, states: Array[State]) -> Array[State]:
        opt_policy, opt_q, beta = self.net(self.penv.extract(states))
        options, is_new_options = self.sample_options(opt_q, beta, self.prev_options)
        policy = opt_policy[self.worker_indices, options]
        actions = policy.action().squeeze().cpu().numpy()
        next_states, rewards, done, info = self.penv.step(actions)
        self.episode_length += 1
        self.rewards += rewards
        self.report_reward(done, info)
        self.storage.push(next_states, rewards, done, policy=policy, value=opt_q)
        self.storage.push_options(options, is_new_options, self.opt_explorer.epsilon)
        return next_states

    @torch.no_grad()
    def _next_value(self, states: Array[State]) -> Tensor:
        opt_q = self.net.opt_q(self.penv.extract(states))
        current_opt_q = opt_q[self.worker_indices, self.prev_options]
        eps = self.opt_explorer.epsilon
        next_opt_q = (1 - eps) * opt_q.max(dim=-1)[0] + eps * opt_q.mean(-1)
        return torch.where(self.prev_is_new_options, next_opt_q, current_opt_q)

    def nstep(self, states: Array[State]) -> Array[State]:
        for _ in range(self.config.nsteps):
            states = self._one_step(states)

        next_value = self._next_value(states)
        self.storage.calc_returns(
            next_value, self.config.discount_factor, self.config.opt_delib_cost,
        )

        prev_options, options = self.storage.batch_options()
        adv = self.storage.advs[:-1].flatten()
        beta_adv = self.storage.beta_adv.flatten()
        ret = self.storage.returns[:-1].flatten()
        masks = self.storage.batch_masks()

        opt_policy, opt_q, beta = self.net(self.storage.batch_states(self.penv))
        policy = opt_policy[self.batch_indices, prev_options]
        batch_actions = self.storage.batch_actions()
        policy.set_action(batch_actions)

        policy_loss = -(policy.log_prob() * adv).mean()
        term_prob = beta[self.batch_indices, prev_options].dist.probs
        beta_adv += self.config.opt_delib_cost + self.config.opt_beta_adv_merginal
        beta_loss = term_prob.mul(masks).mul(beta_adv).mean()
        value_loss = (opt_q[self.batch_indices, options] - ret).pow(2).mean()
        entropy = policy.entropy().mean()

        self.optimizer.zero_grad()
        (
            policy_loss
            + beta_loss
            + self.config.value_loss_weight * 0.5 * value_loss
            - self.config.entropy_weight * entropy
        ).backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
        self.optimizer.step()

        self.network_log(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            beta_loss=beta_loss.item(),
            entropy=entropy.item(),
        )
        self.storage.reset()
        return states
