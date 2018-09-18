import numpy as np
from numpy import ndarray
from torch import nn, Tensor, torch
from typing import Tuple
from .base import Agent
from ..config import Config
from ..net.value_net import ValueNet
from ..explore import Greedy


class DqnAgent(Agent):
    def __init__(self, config: Config) -> None:
        self.net = config.value_net()
        self.target_net = config.value_net()
        self.optimizer = config.optimizer(self.net.parameters())
        self.criterion = nn.MSELoss()
        self.policy = config.explorer(self.net)
        self.total_steps = 0
        self.replay = config.replay_buffer()
        self.env = config.env
        self.config = config
        self.batch_indices = torch.arange(
            config.batch_size,
            device=self.config.device(),
            dtype=torch.long
        )

    def members_to_save(self) -> Tuple[str, ...]:
        return "net", "target_net"

    def episode(self, train: bool = True) -> None:
        if not train:
            self.policy = Greedy(self.net)
        total_reward = 0.0
        steps = 0
        self.env.seed(self.config.seed)
        state = self.env.reset()
        while True:
            state, reward, done = self.step(state, train=train)
            steps += 1
            self.total_steps += 1
            total_reward += reward
            if done:
                break
        print(total_reward)

    def step(self, state: ndarray, train: bool = True) -> Tuple[ndarray, float, bool]:
        train_started = self.total_steps > self.config.train_start
        if not train or train_started:
            action = self.policy.select_action(state)
        else:
            action = np.random.randint(self.env.action_dim)
        next_state, reward, done, _ = self.env.step(action)
        if not train:
            return next_state, reward, done
        self.replay.append(state, action, reward, next_state, done)
        if not train_started:
            return next_state, reward, done
        observation = self.replay.sample(self.config.batch_size)
        states, actions, rewards, next_states, is_terms = map(np.asarray, zip(*observation))
        q_next = self.target_net(self.config.wrap_states(next_states)).detach()
        if self.config.double_q:
            best_actions = torch.argmax(dim=-1)
            q_next = q_next[self.batch_indices, best_actions]
        else:
            q_next, _ = q_next.max(1)
        q_next *= self.config.device.tensor(1.0 - is_terms)
        q_next *= self.config.discount_factor
        q_next += self.config.device.tensor(rewards)
        q_current = self.net(self.config.wrap_states(states))
        q_current = q_current[self.batch_indices, actions]
        loss = self.config.loss(q_current, q_next)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return next_state, reward, done
