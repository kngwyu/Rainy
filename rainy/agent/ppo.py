import torch
from torch import nn, Tensor
from .a2c import A2cAgent
from .nstep_common import FeedForwardSampler
from ..config import Config
from ..envs import State


class PPOAgent(A2cAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.net = config.net('actor-critic')
        self.optimizer = config.optimizer(self.net.parameters())
        self.loss_reporter = {'p': 0.0, 'v': 0.0, 'e': 0.0}

    def _value_loss(self, value: Tensor, old_value: Tensor, returns: Tensor) -> Tensor:
        """Clip value function loss.
        I don't know about this so much but baselines does so (>_<).
        """
        unclipped_loss = (returns - value).pow(2)
        if not self.config.ppo_value_clip:
            return unclipped_loss
        clip_eps = self.config.ppo_clip
        value_clipped = old_value + (value - old_value).clamp(-clip_eps, clip_eps)
        clipped_loss = (value_clipped - returns).pow(2)
        return torch.max(unclipped_loss, clipped_loss)

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
        clip_eps = self.config.ppo_clip
        for _ in range(self.config.ppo_epochs):
            sampler = FeedForwardSampler(
                self.storage,
                self.penv,
                self.config.ppo_minibatch_size,
                adv_noramalize_eps=self.config.adv_normalize_eps,
            )
            for batch in sampler:
                policy, value = self.net(batch.states)
                policy.set_action(batch.actions)
                prob_ratio = torch.exp(policy.lob_prob() - batch.old_log_probs)
                surr1 = prob_ratio * batch.advantages
                surr2 = prob_ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * batch.advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                value_loss = self._value_loss(value, batch.values).mean()
                entropy_loss = policy.entropy().mean()
                self.optimizer.zero_grad()
                (policy_loss
                 + self.config.value_loss_weight * value_loss
                 - self.config.entropy_loss_weight * entropy_loss).backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
                self.optimizer.step()
                # loss reporting will be implemented in the future, maybe
                self.loss_reporter['p'] += policy_loss.item()
                self.loss_reporter['v'] += value_loss.item()
                self.loss_reporter['e'] += entropy_loss.item()
        self.storage.reset()
        return states
