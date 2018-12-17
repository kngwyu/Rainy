import numpy as np
import torch
from torch import nn
from typing import Tuple
from .base import OneStepAgent
from ..config import Config
from ..envs import Action, State
from ..replay import DqnReplayFeed


class DqnAgent(OneStepAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.net = config.net('value')
        self.target_net = config.net('value')
        self.optimizer = config.optimizer(self.net.parameters())
        self.criterion = nn.MSELoss()
        self.policy = config.explorer(self.net)
        self.replay = config.replay_buffer()
        assert self.replay.feed == DqnReplayFeed
        self.batch_indices = torch.arange(
            config.replay_batch_size,
            device=self.config.device.unwrapped,
            dtype=torch.long
        )

    def members_to_save(self) -> Tuple[str, ...]:
        return "net", "target_net", "policy", "total_steps"

    def eval_action(self, state: State) -> Action:
        return self.net.action_values(state).detach().argmax().item()

    def step(self, state: State) -> Tuple[State, float, bool, dict]:
        train_started = self.total_steps > self.config.train_start
        if train_started:
            action = self.policy.select_action(self.env.state_to_array(state))
        else:
            action = self.random_action()
        next_state, reward, done, info = self.env.step(action)
        self.replay.append(state, action, reward, next_state, done)
        if train_started:
            self._train()
        return next_state, reward, done, info

    def _train(self) -> None:
        obs = self.replay.sample(self.config.replay_batch_size)
        obs = [ob.to_ndarray(self.env.state_to_array) for ob in obs]
        states, actions, rewards, next_states, done = map(np.asarray, zip(*obs))
        q_next = self.target_net(next_states).detach()
        if self.config.double_q:
            # Here supposes action_values is replay_batch_size×(action_dim) array
            action_values = self.net.action_values(next_states, nostack=True).detach()
            q_next = q_next[self.batch_indices, action_values.argmax(dim=-1)]
        else:
            q_next, _ = q_next.max(1)
        q_next *= self.config.device.tensor(1.0 - done) * self.config.discount_factor
        q_next += self.config.device.tensor(rewards)
        q_current = self.net(states)[self.batch_indices, actions]
        loss = self.criterion(q_current, q_next)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.report_loss(value_loss=loss.item())
        if self.total_steps % self.config.sync_freq == 0:
            self.sync_target_net()

    def sync_target_net(self) -> None:
        self.target_net.load_state_dict(self.net.state_dict())
