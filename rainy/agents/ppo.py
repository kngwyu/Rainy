"""
This module has an implementation of PPO, which is described in
- Proximal Policy Optimization Algorithms
  - URL: https://arxiv.org/abs/1707.06347
"""
from abc import ABC

import torch
from torch import Tensor

from ..config import Config
from ..envs import State
from ..lib import mpi
from ..lib.rollout import RolloutSampler
from ..net import Policy
from ..prelude import Array
from .a2c import A2CAgent


class PPOLossMixIn(ABC):
    clip_eps: float
    config: Config

    def _proximal_policy_loss(
        self, policy: Policy, advantages: Tensor, old_log_probs: Tensor
    ) -> Tensor:
        prob_ratio = torch.exp(policy.log_prob() - old_log_probs)
        surr1 = prob_ratio * advantages
        surr2 = prob_ratio.clamp(1.0 - self.clip_eps, 1.0 + self.clip_eps) * advantages
        return -torch.min(surr1, surr2).mean()

    def _value_loss(self, value: Tensor, old_value: Tensor, returns: Tensor) -> Tensor:
        """Clip value function loss.
        OpenAI baselines says it reduces variability during Critic training...
        but I'm not sure.
        """
        unclipped_loss = (value - returns).pow(2) * 0.5
        if not self.config.ppo_value_clip:
            return unclipped_loss.mean()
        value_clipped = old_value + (value - old_value).clamp(
            -self.clip_eps, self.clip_eps
        )
        clipped_loss = (value_clipped - returns).pow(2) * 0.5
        return torch.max(unclipped_loss, clipped_loss).mean()


class PPOAgent(A2CAgent, PPOLossMixIn):
    SAVED_MEMBERS = "net", "clip_eps", "clip_cooler", "optimizer"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.clip_cooler = config.clip_cooler()
        self.clip_eps = config.ppo_clip
        self.num_updates = self.config.ppo_epochs * self.config.ppo_num_minibatches
        mpi.setup_models(self.net)
        self.optimizer = mpi.setup_optimizer(self.optimizer)

    def train(self, last_states: Array[State]) -> None:
        with torch.no_grad():
            next_value = self.net.value(*self._network_in(last_states))

        if self.config.use_gae:
            self.storage.calc_gae_returns(
                next_value, self.config.discount_factor, self.config.gae_lambda,
            )
        else:
            self.storage.calc_ac_returns(next_value, self.config.discount_factor)
        p, v, e = 0.0, 0.0, 0.0
        sampler = RolloutSampler(
            self.storage,
            self.penv,
            self.config.ppo_minibatch_size,
            adv_normalize_eps=self.config.adv_normalize_eps,
        )
        for _ in range(self.config.ppo_epochs):
            for batch in sampler:
                policy, value, _ = self.net(batch.states, batch.rnn_init, batch.masks)
                policy.set_action(batch.actions)
                policy_loss = self._proximal_policy_loss(
                    policy, batch.advantages, batch.old_log_probs
                )
                value_loss = self._value_loss(value, batch.values, batch.returns)
                entropy_loss = policy.entropy().mean()
                self.optimizer.zero_grad()
                (
                    policy_loss
                    + self.config.value_loss_weight * 0.5 * value_loss
                    - self.config.entropy_weight * entropy_loss
                ).backward()
                mpi.clip_and_step(
                    self.net.parameters(), self.config.grad_clip, self.optimizer
                )
                p, v, e = (
                    p + policy_loss.item(),
                    v + value_loss.item(),
                    e + entropy_loss.item(),
                )

        self.lr_cooler.lr_decay(self.optimizer)
        self.clip_eps = self.clip_cooler()
        self.storage.reset()

        p, v, e = (x / self.num_updates for x in (p, v, e))
        self.network_log(policy_loss=p, value_loss=v, entropy_loss=e)
