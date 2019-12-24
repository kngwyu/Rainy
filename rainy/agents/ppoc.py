"""
This module has an implementation of
PPOC(Proximal Policy Option Critic), which is described in
- Learnings Options End-to-End for Continuous ActionTasks
  - https://arxiv.org/abs/1712.00004
"""

import torch
from torch import Tensor
from typing import NamedTuple, Optional
from .aoc import AOCRolloutStorage, AOCAgent
from ..lib.rollout import RolloutSampler
from ..lib import mpi
from ..config import Config
from ..envs import ParallelEnv, State
from ..net import Policy
from ..prelude import Array, Index


class RolloutBatch(NamedTuple):
    states: Tensor
    actions: Tensor
    masks: Tensor
    returns: Tensor
    opt_q_values: Tensor
    old_log_probs: Tensor
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

    def _make_batch(self, i: Index) -> RolloutBatch:
        return RolloutBatch(
            self.states[i],
            self.actions[i],
            self.masks[i],
            self.returns[i],
            self.values[i],
            self.old_log_probs[i],
            self.advantages[i],
            self.beta_advantages[i],
            self.prev_options[i],
            self.options[i],
        )


class PPOCAgent(AOCAgent):
    SAVED_MEMBERS = "net", "clip_eps", "clip_cooler", "optimizer"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.clip_cooler = config.clip_cooler()
        self.clip_eps = config.ppo_clip
        self.num_updates = self.config.ppo_epochs * self.config.ppo_num_minibatches
        mpi.setup_models(self.net)
        self.optimizer = mpi.setup_optimizer(self.optimizer)
        self.batch_indices = config.device.indices(config.ppo_minibatch_size)

    def _policy_loss(
        self, policy: Policy, advantages: Tensor, old_log_probs: Tensor
    ) -> Tensor:
        prob_ratio = torch.exp(policy.log_prob() - old_log_probs)
        surr1 = prob_ratio * advantages
        surr2 = prob_ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        return -torch.min(surr1, surr2).mean()

    def _value_loss(
        self, opt_q: Tensor, options: Tensor, old_opt_q: Tensor, returns: Tensor
    ) -> Tensor:
        value = opt_q[self.batch_indices, options]
        old_value = old_opt_q[self.batch_indices, options]
        unclipped_loss = (value - returns).pow(2)
        if not self.config.ppo_value_clip:
            return unclipped_loss.mean()
        value_clipped = old_value + (value - old_value).clamp(
            -self.clip_eps, self.clip_eps
        )
        clipped_loss = (value_clipped - returns).pow(2)
        return torch.max(unclipped_loss, clipped_loss).mean()

    def nstep(self, states: Array[State]) -> Array[State]:
        for _ in range(self.config.nsteps):
            states = self._one_step(states)

        next_v = self._next_value(states)
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

        sampler = PPOCSampler(
            self.storage,
            self.penv,
            self.config.ppo_minibatch_size,
            adv_normalize_eps=self.config.adv_normalize_eps,
        )

        p, v, b, e = 0.0, 0.0, 0.0, 0.0
        for _ in range(self.config.ppo_epochs):
            for batch in sampler:
                opt_policy, opt_q, beta = self.net(batch.states)
                policy = opt_policy[self.batch_indices, batch.prev_options]
                policy.set_action(batch.actions)
                policy_loss = self._policy_loss(
                    policy, batch.advantages, batch.old_log_probs
                )
                value_loss = self._value_loss(
                    opt_q, batch.options, batch.opt_q_values, batch.returns
                )
                entropy_loss = policy.entropy().mean()

                term_prob = beta[self.batch_indices, batch.prev_options].dist.probs
                beta_adv = (
                    batch.beta_advantages
                    + self.config.opt_delib_cost
                    + self.config.opt_beta_adv_merginal
                )
                beta_loss = term_prob.mul(batch.masks * beta_adv).mean()
                self.optimizer.zero_grad()
                (
                    policy_loss * beta_loss
                    + self.config.value_loss_weight * 0.5 * value_loss
                    - self.config.entropy_weight * entropy_loss
                ).backward()
                mpi.clip_and_step(
                    self.net.parameters(), self.config.grad_clip, self.optimizer
                )

                p, v, b, e = (
                    p + policy_loss.item(),
                    b + beta_loss.item(),
                    v + value_loss.item(),
                    e + entropy_loss.item(),
                )

        p, v, b, e = (x / self.num_updates for x in (p, v, b, e))
        self.network_log(
            policy_loss=p, value_loss=v, beta_loss=b, entropy=e,
        )
        self.storage.reset()
        return states
