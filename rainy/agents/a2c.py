import numpy as np
import torch
from torch import nn
from typing import Tuple
from .base import NStepParallelAgent
from ..config import Config
from ..prelude import Action, Array, State


class A2cAgent(NStepParallelAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.net = config.net('actor-critic')
        self.optimizer = config.optimizer(self.net.parameters())
        self.lr_cooler = config.lr_cooler(self.optimizer.param_groups[0]['lr'])

    def members_to_save(self) -> Tuple[str, ...]:
        return ("net",)

    def eval_action(self, state_: Array) -> Action:
        state = self.env.state_to_array(state_)
        if len(state.shape) == len(self.net.state_dim):
            # treat as batch_size == 1
            state = np.stack([state])
        with torch.no_grad():
            policy = self.net.policy(state)
        if self.config.eval_deterministic:
            return policy.best_action().squeeze().cpu().numpy()
        else:
            return policy.action().squeeze().cpu().numpy()

    def eval_action_parallel(self, states: Array[State]) -> Array[Action]:
        with torch.no_grad():
            policy = self.net.policy(self.penv.states_to_array(states))
        if self.config.eval_deterministic:
            return policy.best_action().squeeze().cpu().numpy()
        else:
            return policy.action().squeeze().cpu().numpy()

    def _one_step(self, states: Array[State]) -> Array[State]:
        with torch.no_grad():
            policy, value = self.net(self.penv.states_to_array(states))
        next_states, rewards, done, info = self.penv.step(policy.action().squeeze().cpu().numpy())
        self.episode_length += 1
        self.rewards += rewards
        self.report_reward(done, info)
        self.storage.push(next_states, rewards, done, policy=policy, value=value)
        return next_states

    def _step_optimizer(self) -> None:
        nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.lr_cooler.lr_decay(self.optimizer)

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

        policy, value = self.net(self.storage.batch_states(self.penv))
        policy.set_action(self.storage.batch_actions())

        advantage = self.storage.batch_returns() - value
        policy_loss = -(policy.log_prob() * advantage.detach()).mean()
        value_loss = advantage.pow(2).mean()
        entropy_loss = policy.entropy().mean()
        self.optimizer.zero_grad()
        (policy_loss
         + self.config.value_loss_weight * value_loss
         - self.config.entropy_weight * entropy_loss).backward()
        self._step_optimizer()
        self.report_loss(
            policy_loss=policy_loss.item(),
            value_loss=value_loss.item(),
            entropy_loss=entropy_loss.item(),
        )
        self.storage.reset()
        return states