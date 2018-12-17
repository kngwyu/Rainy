import torch
from torch import nn, Tensor
from .a2c import A2cAgent
from .nstep_common import FeedForwardSampler, lr_decay
from ..config import Config
from ..envs import State
from ..net import Policy
from ..util.typehack import Array


class PpoAgent(A2cAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.net = config.net('actor-critic')
        self.optimizer = config.optimizer(self.net.parameters())
        self.cooler = config.lr_cooler(self.optimizer.param_groups[0]['lr'])
        self.loss_reporter = {'p': 0.0, 'v': 0.0, 'e': 0.0}

    def _policy_loss(self, policy: Policy, advantages: Tensor, old_log_probs: Tensor) -> Tensor:
        clip_eps = self.config.ppo_clip
        prob_ratio = torch.exp(policy.log_prob() - old_log_probs)
        surr1 = prob_ratio * advantages
        surr2 = prob_ratio.clamp(1.0 - clip_eps, 1.0 + clip_eps) * advantages
        return -torch.min(surr1, surr2).mean()

    def _value_loss(self, value: Tensor, old_value: Tensor, returns: Tensor) -> Tensor:
        """Clip value function loss.
        I don't know about this so much but baselines does so (>_<).
        """
        unclipped_loss = (value - returns).pow(2)
        if not self.config.ppo_value_clip:
            return unclipped_loss.mean()
        clip_eps = self.config.ppo_clip
        value_clipped = old_value + (value - old_value).clamp(-clip_eps, clip_eps)
        clipped_loss = (value_clipped - returns).pow(2)
        return torch.max(unclipped_loss, clipped_loss).mean()

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
                # loss reporting will be implemented in the future, maybe
                self.loss_reporter['p'] += policy_loss.item()
                self.loss_reporter['v'] += value_loss.item()
                self.loss_reporter['e'] += entropy_loss.item()
        self.loss_reporter = {'p': 0.0, 'v': 0.0, 'e': 0.0}
        if self.config.lr_decay:
            lr_decay(self.optimizer, self.cooler)
        self.storage.reset()
        return states
