import numpy as np
from numpy import ndarray
import torch
from torch import nn
from typing import Tuple
from .base import Agent
from ..config import Config
from ..env_ext import Action


class DqnAgent(Agent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.net = config.value_net()
        self.target_net = config.value_net()
        self.optimizer = config.optimizer(self.net.parameters())
        self.criterion = nn.MSELoss()
        self.policy = config.explorer(self.net)
        self.replay = config.replay_buffer()
        self.batch_indices = torch.arange(
            config.batch_size,
            device=self.config.device(),
            dtype=torch.long
        )

    def members_to_save(self) -> Tuple[str, ...]:
        return "net", "target_net", "policy"

    def best_action(self, state: ndarray) -> Action:
        action_values = self.net.action_values(state).detach()
        # Here supposes action_values is 1×(action_dim) array
        return action_values.argmax()

    def step(self, state: ndarray) -> Tuple[ndarray, float, bool]:
        train_started = self.total_steps > self.config.train_start
        if train_started:
            action = self.policy.select_action(state)
        else:
            action = np.random.randint(self.env.action_dim)
        next_state, reward, done, _ = self.env.step(action)
        self.replay.append(state, action, reward, next_state, done)
        if not train_started:
            return next_state, reward, done
        observation = self.replay.sample(self.config.batch_size)
        states, actions, rewards, next_states, is_terms = map(np.asarray, zip(*observation))
        q_next = self.target_net(next_states).detach()
        if self.config.double_q:
            # Here supposes action_values is batch_size×(action_dim) array
            action_values = self.net.action_values(next_states, nostack=True).detach()
            q_next = q_next[self.batch_indices, action_values.argmax(dim=-1)]
        else:
            q_next, _ = q_next.max(1)
        q_next *= self.config.device.tensor(1.0 - is_terms) * self.config.discount_factor
        q_next += self.config.device.tensor(rewards)
        q_current = self.net(states)[self.batch_indices, actions]
        loss = self.criterion(q_current, q_next)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        if self.total_steps % self.config.sync_freq == 0:
            self.sync_target_net()
        return next_state, reward, done

    def sync_target_net(self) -> None:
        self.target_net.load_state_dict(self.net.state_dict())
