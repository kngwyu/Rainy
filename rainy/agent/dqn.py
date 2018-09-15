import numpy as np
from torch import nn, Tensor, torch
from .base import Agent
from ..config import Config
from ..net.value_net import ValueNet
from ..explore import Greedy

class DqnAgent(Agent):
    def __init__(self, config: Config):
        self.net = config.value_net()
        self.target_net = config.value_net()
        self.optimizer = config.gen_optimizer(self.net.parameters())
        self.criterion = nn.MSELoss()
        self.policy = config.get_explorer(self.net)
        self.total_steps = 0
        self.replay = config.replay_buffer()
        self.env = config.env()
        self.config = config

    def members_to_save(self):
        return "net", "target_net"

    def episode(self):
        pass

    def episode(self, train: bool = True):
        if not train:
            self.policy = Greedy(self.net)
        total_reward = 0.0
        steps = 0
        self.env.seed(self.config.seed)
        state = self.env.reset()
        while True:
            action = self.policy.select_action(state)
            next_state, reward, done, _ = self.env.step(action)
            total_reward += reward
            if train:
                self.replay.append(state, action, reward, next_state, done)
                self.total_steps += 1
            steps += 1

    def step(self, state: ndarray, train: bool = True):
        train_started = self.total_steps > self.config.train_start
        if train_started:
            action = self.policy.select_action(self.config.wrap_state(state))
        else:
            action = np.random.randint(self.value_net.action_dim)
        next_state, reward, done, _ = self.env.step(state)
        next_state = next_state
        if train:
            self.replay.append(state, action, reward, next_state, done)
            self.total_steps += 1
        if train and train_started:
            observation = self.replay.sample(self.config.batch_size)
            states, actions, rewards, next_states, is_terms = map(np.asarray, zip(*observation))
            next_states = self.wrap_states(next_states)
            q_next = self.target_net(next_states).detach()
            if self.config.double_q:
                best_actions = torch.argmax
