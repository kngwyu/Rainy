"""TD3(Twin Delayed Deep Deterministic Policy Gradient)
Paper: https://arxiv.org/abs/1802.09477
"""
from copy import deepcopy
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from .base import OneStepAgent
from .ddpg import DdpgAgent
from ..config import Config
from ..prelude import Array


class Td3Agent(DdpgAgent):
    def __init__(self, config: Config) -> None:
        OneStepAgent.__init__(self, config)
        self.net = config.net('td3')
        self.target_net = deepcopy(self.net)
        self.actor_opt = config.optimizer(self.net.actor.parameters(), key='actor')
        self.critic_opt = config.optimizer(self.net.critic.parameters(), key='critic')
        self.explorer = config.explorer()
        self.target_explorer = config.explorer(key='target')
        self.eval_explorer = config.explorer(key='eval')
        self.replay = config.replay_buffer()
        self.batch_indices = config.device.indices(config.replay_batch_size)

    @torch.no_grad()
    def _q_target(self, next_states: Array) -> Tensor:
        actions = self.target_net.action(next_states)
        actions = self.env.spec.clip_action(self.target_explorer.add_noise(actions))
        q1, q2 = self.target_net.q_values(next_state, actions)
        return torch.min(q1, q2)

    def _train(self) -> None:
        obs = self.replay.sample(self.config.replay_batch_size)
        obs = [ob.to_ndarray(self.env.extract) for ob in obs]
        states, actions, rewards, next_states, done = map(np.asarray, zip(*obs))
        mask = self.config.device.tensor(1.0 - done)
        q_next = self._q_next(next_states)
        q_target = q_next.squeeze_() \
                         .mul_(mask * self.config.discount_factor) \
                         .add_(self.config.device.tensor(rewards))
        q1, q2 = self.net.q_values(states, actions)

        #  Backward critic loss
        critic_loss = F.mse_loss(q1.squeeze_(), q_target) + F.mse_loss(q2.squeeze_(), q_target)
        self._backward(critic_loss, self.net.critic, self.critic_opt)

        #  Delayed policy update
        if (self.update_steps + 1) % self.config.policy_update_freq != 0:
            return
        action = self.net.action(states)
        policy_loss = -self.net.q_value(states, action).mean()
        self._backward(policy_loss, self.net.actor, self.actor_opt)

        #  Update target network
        self.target_net.soft_update(self.net, self.config.soft_update_coef)
