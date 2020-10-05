"""
This module has an implementation of
PPOC(Proximal Policy Option Critic), which is described in
- Learnings Options End-to-End for Continuous ActionTasks
  - https://arxiv.org/abs/1712.00004
"""

from typing import NamedTuple, Optional, Tuple

import torch
from torch import BoolTensor, LongTensor, Tensor

from ..config import Config
from ..envs import ParallelEnv
from ..lib import mpi
from ..lib.rollout import RolloutSampler
from ..net.policy import BernoulliPolicy, CategoricalPolicy, Policy
from ..prelude import Action, Array, Index, State
from .aoc import AOCAgent, AOCRolloutStorage
from .ppo import PPOLossMixIn


class RolloutBatch(NamedTuple):
    states: Tensor
    actions: Tensor
    masks: Tensor
    returns: Tensor
    old_log_probs: Tensor
    old_log_probs_mu: Tensor
    advantages: Tensor
    beta_advantages: Tensor
    prev_options: Tensor
    options: Tensor


class PPOCSampler(RolloutSampler):
    def __init__(
        self,
        storage: AOCRolloutStorage,
        penv: ParallelEnv,
        minibatch_size: int,
        adv_normalize_eps: Optional[float] = None,
    ) -> None:
        torch.stack(storage.values, dim=0, out=storage.batch_values)
        super().__init__(
            storage, penv, minibatch_size, adv_normalize_eps=adv_normalize_eps
        )
        self.prev_options, self.options = storage.batch_options()
        self.beta_advantages = storage.beta_adv.flatten()
        mu_logits = torch.cat([p.dist.logits for p in storage.muos])
        mu = CategoricalPolicy(logits=mu_logits)
        mu.set_action(self.options)
        self.old_log_probs_mu = mu.log_prob()

    def _make_batch(self, i: Index) -> RolloutBatch:
        return RolloutBatch(
            self.states[i],
            self.actions[i],
            self.masks[i],
            self.returns[i],
            self.old_log_probs[i],
            self.old_log_probs_mu[i],
            self.advantages[i],
            self.beta_advantages[i],
            self.prev_options[i],
            self.options[i],
        )


class PPOCAgent(AOCAgent, PPOLossMixIn):
    SAVED_MEMBERS = "net", "clip_eps", "clip_cooler", "optimizer"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if not hasattr(self.net, "mu_head"):
            import warnings

            warnings.warn("PPOC net doesn't have mu_head")
        self.clip_cooler = config.clip_cooler()
        self.clip_eps = config.ppo_clip
        self.num_updates = self.config.ppo_epochs * self.config.ppo_num_minibatches
        mpi.setup_models(self.net)
        self.optimizer = mpi.setup_optimizer(self.optimizer)
        self.batch_indices = config.device.indices(config.ppo_minibatch_size)
        if not config.opt_avg_baseline:
            self.storage._use_mu_for_beta_adv()
        if config.proximal_update_for_mu:
            self._mu_policy_loss = self._proximal_policy_loss
        else:
            self._mu_policy_loss = self._normal_policy_loss

    def _normal_policy_loss(self, policy: Policy, advantages: Tensor, *args,) -> Tensor:
        return -(policy.log_prob() * advantages).mean()

    def _value_loss(self, qo: Tensor, options: Tensor, returns: Tensor) -> Tensor:
        return (qo[self.batch_indices, options] - returns).pow(2).mean()

    @torch.no_grad()
    def _eval_policy(self, states: Array) -> Policy:
        batch_size = states.shape[0]
        pio, _, beta, mu = self.net(states)
        options, _ = self._sample_options(
            mu, beta, self.eval_prev_options[:batch_size], evaluation_phase=True,
        )
        self.eval_prev_options[:batch_size] = options
        return pio, options

    def _sample_options(
        self,
        mu: CategoricalPolicy,
        beta: BernoulliPolicy,
        prev_options: LongTensor,
        evaluation_phase: bool = False,
    ) -> Tuple[LongTensor, BoolTensor]:
        current_beta = beta[self.worker_indices[: prev_options.size(0)], prev_options]
        do_options_end = current_beta.action().bool()
        if evaluation_phase:
            use_new_options = do_options_end
        else:
            is_initial_states = (1.0 - self.storage.masks[-1]).bool()
            use_new_options = do_options_end | is_initial_states
        sampled_options = mu.eval_action(deterministic=evaluation_phase, to_numpy=False)
        options = torch.where(use_new_options, sampled_options, prev_options)
        return options, use_new_options  # type: ignore

    @torch.no_grad()
    def actions(self, states: Array[State]) -> Tuple[Array[Action], dict]:
        pio, value, beta, mu = self.net(self.penv.extract(states))
        options, do_options_end = self._sample_options(mu, beta, self.prev_options)
        pi = pio[self.worker_indices, options]
        actions = pi.action().squeeze().cpu().numpy()
        net_outputs = dict(
            policy=pi,
            value=value,
            options=options,
            opt_terminals=do_options_end,
            mu=mu,
        )
        return actions, net_outputs

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
                next_uo, self.config.discount_factor, self.config.opt_delib_cost
            )

        sampler = PPOCSampler(
            self.storage,
            self.penv,
            self.config.ppo_minibatch_size,
            adv_normalize_eps=self.config.adv_normalize_eps,
        )

        p, v, b, pe, m, me = (0.0,) * 6
        for _ in range(self.config.ppo_epochs):
            for batch in sampler:
                pio, qo, beta, mu = self.net(batch.states)
                # Policy loss
                pi = pio[self.batch_indices, batch.options]
                pi.set_action(batch.actions)
                policy_loss = self._proximal_policy_loss(
                    pi, batch.advantages, batch.old_log_probs,
                )
                # Value loss
                value_loss = self._value_loss(qo, batch.options, batch.returns)
                # Beta loss
                term_prob = beta[self.batch_indices, batch.prev_options].dist.probs
                beta_adv = (
                    batch.beta_advantages
                    + self.config.opt_delib_cost
                    + self.config.opt_beta_adv_merginal
                )
                beta_loss = term_prob.mul(batch.masks * beta_adv).mean()
                # Mu loss
                mu.set_action(batch.options)
                mu_loss = self._mu_policy_loss(
                    mu, batch.beta_advantages, batch.old_log_probs_mu
                )
                # Entropy loss
                pe_loss = pi.entropy().mean()
                me_loss = mu.entropy().mean()
                self.optimizer.zero_grad()
                (
                    policy_loss
                    + beta_loss
                    + mu_loss
                    + self.config.value_loss_weight * 0.5 * value_loss
                    - self.config.entropy_weight * pe_loss
                    - self.config.entropy_weight * me_loss
                ).backward()
                mpi.clip_and_step(
                    self.net.parameters(), self.config.grad_clip, self.optimizer
                )

                p, v, b, pe, m, me = (
                    p + policy_loss.item(),
                    b + beta_loss.item(),
                    v + value_loss.item(),
                    m + mu_loss.item(),
                    pe + pe_loss.item(),
                    me + me_loss.item(),
                )

        p, v, b, pe, m, me = (x / self.num_updates for x in (p, v, b, pe, m, me))
        self.network_log(
            policy_loss=p, value_loss=v, beta_loss=b, entropy=pe, mu=m, mu_entropy=me,
        )
        self.storage.reset()
