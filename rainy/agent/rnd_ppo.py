import torch
from torch import nn, Tensor
from typing import Tuple
from .ppo import PpoAgent
from .rollout import FeedForwardSampler, lr_decay
from ..config import Config
from ..envs import State
from ..net import Policy
from ..util.typehack import Array


class RndPpoAgent(PpoAgent):
    """STUB
    """
    def nstep(self, states: Array[State]) -> Array[State]:
        for _ in range(self.config.nsteps):
            states = self._one_step(states)

        with torch.no_grad():
            next_value = self.net.value(self.penv.states_to_array(states))

        if self.config.use_gae:
            gamma, tau = self.config.discount_factor, self.config.gae_tau
            self.storage.calc_gae_returns(next_value, gamma, tau)
        else:
            self.storage.calc_ac_returns(next_value, self.config.discount_factor)
        p, v, e = (0.0, 0.0, 0.0)
        for _ in range(self.config.ppo_epochs):
            sampler = FeedForwardSampler(
                self.storage,
                self.penv,
                self.config.ppo_minibatch_size,
                adv_normalize_eps=self.config.adv_normalize_eps,
            )
            for batch in sampler:
                policy, value = self.net(batch.states)
                policy.set_action(batch.actions)
                policy_loss = self._policy_loss(policy, batch.advantages, batch.old_log_probs)
                value_loss = self._value_loss(value, batch.values, batch.returns)
                entropy_loss = policy.entropy().mean()
                self.optimizer.zero_grad()
                (policy_loss
                 + self.config.value_loss_weight * value_loss
                 - self.config.entropy_weight * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
                self.optimizer.step()
                p, v, e = p + policy_loss.item(), v + value_loss.item(), e + entropy_loss.item()

        lr_decay(self.optimizer, self.lr_cooler)
        self.clip_eps = self.clip_cooler(self.clip_eps)
        self.storage.reset()

        p, v, e = map(lambda x: x / float(self.num_updates), (p, v, e))
        self.report_loss(policy_loss=p, value_loss=v, entropy_loss=e)
        return states
