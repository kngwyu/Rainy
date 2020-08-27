"""
This module has an implementation of A2C-like variant of Termination Critic algorithm,
which is described in
- The Termination Critic
  - https://arxiv.org/abs/1902.09996
"""
from typing import Optional, Tuple

import gym
import numpy as np
import torch
from torch import BoolTensor, LongTensor, Tensor
from torch.nn import functional as F

from ..config import Config
from ..lib.explore import EpsGreedy
from ..net import OptionActorCriticNet, TerminationCriticNet
from ..net.policy import BernoulliPolicy, Policy
from ..prelude import Action, Array, State
from .aoc import AOCRolloutStorage
from .base import A2CLikeAgent, Netout


class TCRolloutStorage(AOCRolloutStorage):
    def calc_ac_returns(self, next_uo: Tensor, gamma: float) -> None:
        self.returns[-1] = next_uo
        rewards = self.device.tensor(self.rewards)
        opt_terminals = self.device.zeros((self.nworkers,), dtype=torch.bool)
        for i in reversed(range(self.nsteps)):
            qo, opt = self.values[i], self.options[i + 1]
            eps = self.epsilons[i]
            vo = (1 - eps) * qo.max(dim=-1)[0] + eps * qo.mean(-1)
            ret_i1 = torch.where(opt_terminals, vo, self.returns[i + 1],)
            self.returns[i] = gamma * self.masks[i + 1] * ret_i1 + rewards[i]
            self.advs[i] = self.returns[i] - qo[self.worker_indices, opt]
            opt_terminals = self.opt_terminals[i + 1]

    def calc_p_target(self, beta_x: Tensor, beta_xf: Tensor) -> Tensor:
        res = self.device.zeros((self.nsteps, self.nworkers))
        beta_x = beta_x.view(self.nsteps, self.nworkers)
        p_xiplus1_xf = beta_xf
        for i in reversed(range(self.nsteps)):
            p_x_xf = (1.0 - beta_x[i]).mul_(p_xiplus1_xf)
            p_x_x = beta_x[i]
            res[i] = torch.where(self.opt_terminals[i + 1], p_x_x, p_x_xf)
            p_xiplus1_xf = res[i]
        return res

    def _prepare_xs(self, xs: Tensor, batch_states: Tensor) -> Tensor:
        state_shape = batch_states.shape[1:]
        states = batch_states.view(self.nsteps, self.nworkers, -1)
        xs_last = xs.view(self.nworkers, -1)
        res = []
        for i in range(self.nsteps):
            opt_terminals = self.opt_terminals[i].unsqueeze(1)
            xs_last = torch.where(opt_terminals, states[i], xs_last)
            res.append(xs_last)
        return torch.cat(res).view(self.nsteps * self.nworkers, *state_shape)

    def _prepare_xf(self, xf: Tensor, batch_states: Tensor) -> Tensor:
        state_shape = batch_states.shape[1:]
        states = batch_states.view(self.nsteps, self.nworkers, -1)
        xf_last = xf.view(self.nworkers, -1)
        res = []
        for i in reversed(range(self.nsteps)):
            opt_terminals = self.opt_terminals[i + 1].unsqueeze(1)
            xf_last = torch.where(opt_terminals, states[i], xf_last)
            res.append(xf_last)
        res.reverse()
        return torch.cat(res).view(self.nsteps * self.nworkers, *state_shape)

    def _prepare_raw_xf(self, batch_states: Array) -> Array:
        states = batch_states.reshape(self.nsteps, self.nworkers, -1)
        xf_last = self.states[-1]
        res = []
        for i in reversed(range(self.nsteps)):
            opt_terminals = self.opt_terminals[i + 1].unsqueeze(1).cpu().numpy()
            xf_last = np.where(opt_terminals, states[i], xf_last)
            res.append(xf_last)
        res.reverse()
        return np.concatenate(res)


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

    EPS = 1e-6
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
        self._eval_masks = self.config.device.ones(self.config.nworkers)
        # Environment specific implementation: use count table to get exact Pμ
        if self.config.tc_exact_pmu:
            if hasattr(self.config.eval_env.unwrapped, "raw_observation_space"):
                self._setup_xf_table(
                    self.config.eval_env.unwrapped.raw_observation_space
                )
                self._do_xf_count = True
            else:
                import warnings

                warnings.warn(
                    "tc_exact_pmu requires the environment has raw_observation_space"
                )
        else:
            self._do_xf_count = False

    def _setup_xf_table(self, space: gym.spaces.Box) -> None:
        """Setup count table to get exact Pμ
        """
        if not np.issubdtype(space.dtype, np.integer):
            raise ValueError("raw_observation_space have to be tabular!")
        range_ = space.high - space.low + 1
        self._xf_table = np.zeros((self.noptions, *range_), dtype=np.int32)

    def eval_reset(self) -> None:
        self.eval_prev_options.fill_(0)
        self.eval_initial_states = None

    def _update_xf_count(self, opt_terminals: Array[bool]) -> None:
        if not opt_terminals.any():
            return
        prev_states = self.storage.states[-1]
        xf = prev_states[opt_terminals]
        opt = self.prev_options.cpu().numpy()[opt_terminals]
        for op, shape in zip(opt, xf):
            self._xf_table[(op, *shape)] += 1

    def _pmu_from_count_impl(self, x: Array) -> Array:
        """Suppose x is (batch, state_dim) array
        """
        batch_size = x.shape[0]
        options = torch.cat(self.storage.options[:-1]).cpu().numpy()
        total = self._xf_table[options].reshape(batch_size, -1).sum(-1)  # (batch, )
        dims = tuple(dim.squeeze() for dim in np.hsplit(x, x.shape[1]))
        specific = self._xf_table[(options,) + dims]
        return specific / (total + self.EPS)

    def _pmu_from_count(self) -> Tuple[Array, Array]:
        x = np.concatenate(self.storage.states[: self.config.nsteps], axis=0)
        pmu_x = self._pmu_from_count_impl(x)
        xf = self.storage._prepare_raw_xf(x)
        pmu_xf = self._pmu_from_count_impl(xf)
        return pmu_x, pmu_xf

    def _reset(self, initial_states: Array[State]) -> None:
        self.storage.set_initial_state(initial_states)
        self.option_initial_states = self.penv.extract(initial_states)
        self._xs_reserved = self.tensor(self.option_initial_states)

    def _sample_options(
        self, qo: Tensor, beta: BernoulliPolicy, evaluate: bool = False,
    ) -> Tuple[LongTensor, BoolTensor]:
        """Sample options by ε-Greedy
        """
        batch_size = qo.size(0)
        if evaluate:
            explorer = self.eval_opt_explorer
            masks = self._eval_masks[:batch_size]
            prev_options = self.eval_prev_options[:batch_size]
        else:
            explorer = self.opt_explorer
            masks = self.storage.masks[-1]
            prev_options = self.prev_options
        current_beta = beta[self.worker_indices, prev_options]
        do_options_end = current_beta.action().bool()
        is_initial_states = (1.0 - masks).bool()
        use_new_options = do_options_end | is_initial_states
        epsgreedy_options = explorer.select_from_value(qo, same_device=True)
        options = torch.where(use_new_options, epsgreedy_options, prev_options)
        return options, do_options_end  # type: ignore

    @torch.no_grad()
    def _eval_policy(self, states: Array) -> Policy:
        opt_policy, qo = self.ac_net(states)
        if self.eval_initial_states is None:
            self.eval_initial_states = states.copy()
        beta = self.tc_net.beta(self.eval_initial_states, states)
        options, _ = self._sample_options(qo, beta, evaluate=True,)
        self.eval_prev_options = options
        return opt_policy[self.worker_indices, options]

    def eval_action(self, state: Array, net_outputs: Optional[Netout] = None) -> Action:
        if state.ndim == len(self.ac_net.state_dim):
            # treat as batch_size == nworkers
            state = np.stack([state] * self.config.nworkers)
        policy = self._eval_policy(state)
        return policy[0].eval_action(self.config.eval_deterministic)

    def eval_action_parallel(self, states: Array) -> Array[Action]:
        policy = self._eval_policy(states)
        return policy.eval_action(self.config.eval_deterministic)

    @property
    def prev_options(self) -> LongTensor:
        return self.storage.options[-1]  # type: ignore

    @property
    def prev_opt_terminals(self) -> BoolTensor:
        return self.storage.opt_terminals[-1]  # type: ignore

    @torch.no_grad()
    def actions(self, states: Array[State]) -> Tuple[Array[Action], dict]:
        x = self.penv.extract(states)
        opt_policy, qo = self.ac_net(x)
        tc_output = self.tc_net(self.option_initial_states, x)
        options, opt_terminals = self._sample_options(qo, tc_output.beta)
        if self._do_xf_count:
            self._update_xf_count(opt_terminals.cpu().numpy())
        policy = opt_policy[self.worker_indices, options]
        actions = policy.action().squeeze().cpu().numpy()
        net_outputs = dict(
            policy=policy,
            value=qo,
            options=options,
            opt_terminals=opt_terminals,
            epsilon=1.0 if self.config.opt_avg_baseline else self.opt_explorer.epsilon,
        )
        return actions, net_outputs

    def _one_step(self, states: Array[State]) -> Array[State]:
        actions, net_outputs = self.actions(states)
        transition = self.penv.step(actions).map_r(lambda r: r * self.reward_scale)
        opt_terminals = net_outputs["opt_terminals"].cpu().numpy()
        states = self.penv.extract(transition.states)
        self.option_initial_states[opt_terminals] = states[opt_terminals]
        self.storage.push(*transition[:3], **net_outputs)
        self.returns += transition.rewards
        self.episode_length += 1
        self._report_reward(transition.terminals, transition.infos)
        return transition.states

    @torch.no_grad()
    def _next_uo(self, states: Tensor, beta_next: Tensor) -> Tensor:
        qo = self.ac_net.qo(states)
        qo_current = qo[self.worker_indices, self.prev_options]
        eps = self.opt_explorer.epsilon
        vo = (1 - eps) * qo.max(dim=-1)[0] + eps * qo.mean(-1)
        # Uo(s, o) = (1.0 - β) Qo(s, o) + β Vo(s)
        return (1.0 - beta_next) * qo_current + beta_next * vo

    def train(self, last_states: Array[State]) -> None:
        """Train the agent using N step trajectory
        """
        # N: Number of Steps W: Number of workers O: Number of options
        N, W = self.config.nsteps, self.config.nworkers
        last_states = self.tensor(self.penv.extract(last_states))

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
        p_x_xs = p_x_xs.detach_()[self.batch_indices, prev_options]
        p_mu_x = p_mu_x[self.batch_indices, prev_options]

        beta_xf = beta_xf.dist.probs.detach_()[self.batch_indices, prev_options]
        p_xf_x = p_xf_x[self.batch_indices, prev_options]
        if self._do_xf_count:
            p_mu_x_count, p_mu_xf_count = self._pmu_from_count()
            beta_adv_raw = calc_beta_adv(
                self.tensor(p_mu_x_count), p_x_xs, self.tensor(p_mu_xf_count), p_xf_xs
            )
        else:
            p_mu_xf_avg = (
                p_mu_xf.detach()
                .view(N, W, -1)
                .mean(0)
                .repeat(N, 1)[self.batch_indices, prev_options]
            )
            beta_adv_raw = calc_beta_adv(p_mu_x.detach(), p_x_xs, p_mu_xf_avg, p_xf_xs)
        p_mu_xf = p_mu_xf[self.batch_indices, prev_options]
        baseline = baseline[self.batch_indices, prev_options]

        next_uo = self._next_uo(last_states, beta_xf[-self.config.nworkers :])
        self.storage.calc_ac_returns(next_uo, self.config.discount_factor)

        beta_adv = beta_adv_raw - baseline.detach()
        beta_loss = -(beta_x.dist.logits * beta_x.dist.probs.detach() * beta_adv).mean()
        beta_xf_averaged = beta_xf.view(N, W).mean(dim=0)
        p_target = self.storage.calc_p_target(
            beta_x.dist.probs.detach(), beta_xf_averaged
        )
        p_loss = F.mse_loss(p_xf_x, p_target.flatten())
        pmu_loss = F.mse_loss(p_mu_xf, p_xf_x.detach()) + F.mse_loss(p_mu_x, p_x_xs)
        baseline_loss = F.mse_loss(baseline, beta_adv_raw)
        tc_loss = beta_loss + p_loss + pmu_loss * 0.5 + baseline_loss
        self._backward(tc_loss, self.tc_optimizer, self.tc_net.parameters())

        policy, qo = self.ac_net(x)
        policy = policy[self.batch_indices, options]
        policy.set_action(self.storage.batch_actions())
        policy_loss = -(policy.log_prob() * self.storage.advs[:-1].flatten()).mean()

        qo_ = qo[self.batch_indices, options]
        value_loss = (qo_ - self.storage.returns[:-1].flatten()).pow(2).mean()
        entropy = policy.entropy().mean()

        ac_loss = (
            policy_loss
            + self.config.value_loss_weight * 0.5 * value_loss
            - self.config.entropy_weight * entropy
        )
        self._backward(ac_loss, self.optimizer, self.ac_net.parameters())

        self.network_log(
            policy_loss=policy_loss.item(),
            value=qo_.detach_().mean().item(),
            value_loss=value_loss.item(),
            beta=beta_x.dist.probs.detach_().mean().item(),
            pmu=p_mu_xf.mean().item(),
            beta_loss=beta_loss.item(),
            entropy=entropy.item(),
            p_loss=p_loss.item(),
            pmu_loss=pmu_loss.item(),
            baseline_loss=baseline_loss.item(),
        )
        self.storage.reset()
        self._xs_reserved = self.tensor(self.option_initial_states)
