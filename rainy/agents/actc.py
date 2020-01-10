"""
This module has an implementation of A2C-like variant of Termination Critic algorithm,
which is described in
- The Termination Critic
  - https://arxiv.org/abs/1902.09996
"""
import numpy as np
import torch
from torch import BoolTensor, LongTensor, Tensor
from torch.nn import functional as F
from typing import Optional, Tuple
from .base import A2CLikeAgent, Netout
from ..config import Config
from ..lib.explore import EpsGreedy
from ..lib.rollout import RolloutStorage
from ..net import OptionActorCriticNet, TerminationCriticNet
from ..net.policy import BernoulliPolicy, Policy
from ..prelude import Action, Array, State
from ..utils import Device


class TCRolloutStorage(RolloutStorage[State]):
    def __init__(
        self, nsteps: int, nworkers: int, device: Device, num_options: int,
    ) -> None:
        super().__init__(nsteps, nworkers, device)
        self.options = [self.device.zeros(self.nworkers, dtype=torch.long)]
        self.is_new_options = [self.device.ones(self.nworkers, dtype=torch.bool)]
        self.noptions = num_options
        self.worker_indices = self.device.indices(self.nworkers)

    def reset(self) -> None:
        super().reset()
        self.options = [self.options[-1]]
        self.is_new_options = [self.is_new_options[-1]]

    def push(
        self, *args, options: LongTensor, is_new_options: Tensor, **kwargs,
    ) -> None:
        super().push(*args, **kwargs)
        self.options.append(options)
        self.is_new_options.append(is_new_options)

    def batch_options(self) -> Tuple[Tensor, Tensor]:
        batched = torch.cat(self.options, dim=0)
        return batched[: -self.nworkers], batched[self.nworkers :]

    def calc_ac_returns(self, next_value: Tensor, gamma: float) -> None:
        self.returns[-1] = next_value
        rewards = self.device.tensor(self.rewards)
        for i in reversed(range(self.nsteps)):
            self.returns[i] = (
                gamma * self.masks[i + 1] * self.returns[i + 1] + rewards[i]
            )
            opt_q, opt = self.values[i], self.options[i + 1]
            self.advs[i] = self.returns[i] - opt_q[self.worker_indices, opt]

    def calc_p_target(self, beta_x: Tensor, beta_xf: Tensor) -> Tensor:
        res = self.device.zeros((self.nsteps, self.nworkers))
        p_xiplus1_xf = beta_xf
        for i in reversed(range(self.nsteps)):
            p_x_xf = (1.0 - beta_x[i]) * p_xiplus1_xf
            p_x_x = beta_x[i]
            res[i] = torch.where(self.is_new_options[i + 1], p_x_x, p_x_xf)
            p_xiplus1_xf = res[i]
        return res

    def _prepare_xs(self, xs: Tensor, batch_states: Tensor) -> Tensor:
        states = batch_states.view(self.nsteps, self.nworkers, *batch_states.shape[1:])
        res = [xs]
        for i in range(1, self.nsteps):
            is_new_options = self.is_new_options[i - 1].view(self.nworkers, 1)
            res.append(torch.where(is_new_options, states[i], res[i - 1]))
        return torch.cat(res)

    def _prepare_xf(self, xf: Tensor, batch_states: Tensor) -> Tensor:
        states = batch_states.view(self.nsteps, self.nworkers, *batch_states.shape[1:])
        res = [xf]
        for i in reversed(range(self.nsteps - 1)):
            is_new_options = self.is_new_options[i].view(self.nworkers, 1)
            res.append(torch.where(is_new_options, states[i], res[-1]))
        res.reverse()
        return torch.cat(res)


def calc_beta_adv(
    p_mu_x: Tensor, p_x_xs: Tensor, p_mu_xf: Tensor, p_xf_xs: Tensor
) -> Tensor:
    P_EPS = 1e-8  # Prevent p=0 -> log(p) = -inf
    adv_p = (p_mu_x + P_EPS).log() - (p_mu_xf + P_EPS).log()
    log_p_ratio = (p_xf_xs * p_mu_x + P_EPS).log() - (p_mu_xf * p_x_xs + P_EPS).log()
    adv_tau = 1.0 - torch.exp(log_p_ratio)
    return adv_p + adv_tau


class ACTCAgent(A2CLikeAgent[State]):
    """ACTC: Actor Critic Termination Critic
    """

    SAVED_MEMBERS = "ac_net", "tc_net", "optimizer", "tc_optimizer"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.ac_net: OptionActorCriticNet = config.net("actor-critic")
        self.tc_net: TerminationCriticNet = config.net("termination-critic")
        self.noptions = self.ac_net.num_options
        self.optimizer = config.optimizer(self.ac_net.parameters())
        self.tc_optimizer = config.optimizer(
            self.tc_net.parameters(), key="termination"
        )
        self.worker_indices = config.device.indices(config.nworkers)
        self.batch_indices = config.device.indices(config.batch_size)
        self.storage = TCRolloutStorage(
            config.nsteps, config.nworkers, config.device, self.noptions
        )
        self.option_initial_states = np.empty(())
        self._xs_reserved = self.config.device.zeros(())
        self.opt_explorer: EpsGreedy = config.explorer()
        self.eval_opt_explorer: EpsGreedy = config.explorer(key="eval")
        if not isinstance(self.opt_explorer, EpsGreedy) or not isinstance(
            self.eval_opt_explorer, EpsGreedy
        ):
            return ValueError("Currently only Epsilon Greedy is supported as Explorer")
        self.eval_prev_options: LongTensor = config.device.zeros(
            config.nworkers, dtype=torch.long
        )
        self.eval_initial_states = None

    def eval_reset(self) -> None:
        self.eval_prev_options.fill_(0)
        self.eval_initial_states = None

    def _update_option_initial_states(
        self, is_new_options: Array[bool], new_states: Array
    ) -> None:
        self.option_initial_states[is_new_options] = new_states[is_new_options]

    def _reset(self, initial_states: Array[State]) -> None:
        self.storage.set_initial_state(initial_states)
        self.option_initial_states = self.penv.extract(initial_states)
        self._xs_reserved = self.tensor(self.option_initial_states)

    def _sample_options(
        self,
        opt_q: Tensor,
        beta: BernoulliPolicy,
        prev_options: LongTensor,
        explorer: Optional[EpsGreedy] = None,
    ) -> Tuple[LongTensor, BoolTensor]:
        """TODO: ε-Greedy is appropriate here?
        """
        if explorer is None:
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
        opt_policy, opt_q = self.ac_net(states)
        if self.eval_initial_states is None:
            self.eval_initial_states = states.copy()
        beta = self.tc_net.beta(self.eval_initial_states, states)
        options, _ = self._sample_options(
            opt_q, beta, self.eval_prev_options, self.eval_opt_explorer
        )
        self.eval_prev_options = options
        return opt_policy[self.worker_indices, options]

    def eval_action(self, state: Array, net_outputs: Optional[Netout] = None) -> Action:
        if len(state.shape) == len(self.ac_net.state_dim):
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
        x = self.penv.extract(states)
        opt_policy, opt_q = self.ac_net(x)
        tc_output = self.tc_net(self.option_initial_states, x)
        options, is_new_options = self._sample_options(
            opt_q, tc_output.beta, self.prev_options
        )
        self._update_option_initial_states(is_new_options.cpu().numpy(), x)
        policy = opt_policy[self.worker_indices, options]
        actions = policy.action().squeeze().cpu().numpy()
        net_outputs = dict(
            policy=policy,
            value=opt_q,
            options=options,
            is_new_options=is_new_options,
            p_x_xs_preds=tc_output.p,
            p_mu_x_pred=tc_output.p_mu,
            beta=tc_output.beta,
        )
        return actions, net_outputs

    @torch.no_grad()
    def _next_value(self, states: Tensor) -> Tuple[Tensor, Tensor]:
        opt_q = self.ac_net.opt_q(states)
        current_opt_q = opt_q[self.worker_indices, self.prev_options]
        eps = self.opt_explorer.epsilon
        next_opt_q = (1 - eps) * opt_q.max(dim=-1)[0] + eps * opt_q.mean(-1)
        return torch.where(self.prev_is_new_options, next_opt_q, current_opt_q)

    def train(self, last_states: Array[State]) -> None:
        """Train the agent using N step trajectory
        """
        # N: Number of Steps W: Number of workers O: Number of options
        N, W = self.config.nsteps, self.config.nworkers
        last_states = self.tensor(self.penv.extract(last_states))
        next_v = self._next_value(last_states)
        self.storage.calc_ac_returns(next_v, self.config.discount_factor)

        prev_options, options = self.storage.batch_options()  # NW
        x = self.storage.batch_states(self.penv)  # NW
        xs = self.storage._prepare_xs(self._xs_reserved, x)  # NW
        xf = self.storage._prepare_xf(last_states, x)  # NW
        with torch.no_grad():
            p_xf_xs = self.tc_net.p(xs, xf)  # NW x O
            p_xf_xs = p_xf_xs[self.batch_indices, prev_options]  # NW

        beta_x, p_x_xs, p_mu_x, _ = self.tc_net(xs, x)
        beta_xf, p_xf_x, p_mu_xf, baseline = self.tc_net(x, xf)

        beta_x = beta_x[self.batch_indices, prev_options]
        bx_p, bx_l = beta_x.dist.probs, beta_x.dist.logits
        p_x_xs = p_x_xs.detach_()[self.batch_indices, prev_options]
        p_mu_x = p_mu_x.detach_()[self.batch_indices, prev_options]

        beta_xf = beta_xf.dist.probs.detach_()[self.batch_indices, prev_options]
        p_xf_x = p_xf_x[self.batch_indices, prev_options]
        with torch.no_grad():
            p_mu_xf_averaged = (
                p_mu_xf.detach()
                .view(N, W, -1)
                .mean(0)
                .repeat(N, 1)[self.batch_indices, prev_options]
            )
        p_mu_xf = p_mu_xf[self.batch_indices, prev_options]
        baseline = baseline[self.batch_indices, prev_options]

        beta_adv = calc_beta_adv(p_mu_x, p_x_xs, p_mu_xf_averaged, p_xf_xs)
        beta_loss = -(bx_l * bx_p * (beta_adv - baseline.detach())).mean()
        beta_xf_averaged = beta_xf.view(N, W).mean(dim=0)
        p_target = self.storage.calc_p_target(
            beta_x.dist.probs.detach(), beta_xf_averaged
        )
        p_loss = (p_target.flatten() - p_xf_x).pow(2).mean()
        pmu_loss = F.mse_loss(p_mu_xf, p_xf_x.detach())
        baseline_loss = F.mse_loss(baseline, beta_adv)
        tc_loss = beta_loss + 0.5 * p_loss + pmu_loss + baseline_loss
        self._backward(tc_loss, self.tc_optimizer, self.tc_net.parameters())

        policy, q = self.ac_net(x)
        policy = policy[self.batch_indices, prev_options]
        policy.set_action(self.storage.batch_actions())
        policy_loss = -(policy.log_prob() * self.storage.advs[:-1].flatten()).mean()
        ret = self.storage.returns[:-1].flatten()
        value_loss = (ret - q[self.batch_indices, options]).pow(2).mean()
        entropy = policy.entropy().mean()

        ac_loss = (
            policy_loss
            + self.config.value_loss_weight * 0.5 * value_loss
            - self.config.entropy_weight * entropy
        )
        self._backward(ac_loss, self.optimizer, self.ac_net.parameters())

        self.network_log(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            beta=beta_x.dist.probs.mean().item(),
            beta_loss=beta_loss.item(),
            entropy=entropy.item(),
            p_loss=p_loss.item(),
            pmu_loss=pmu_loss.item(),
            baseline_loss=baseline_loss.item(),
        )
        self.storage.reset()
        self._xs_reserved = self.tensor(self.option_initial_states)
