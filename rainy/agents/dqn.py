from copy import deepcopy
import numpy as np
import torch
from torch import nn, Tensor
from typing import Tuple
from .base import OneStepAgent
from ..config import Config
from ..replay import DQNReplayFeed
from ..prelude import Action, Array, State


class DQNAgent(OneStepAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        assert self.env.spec.is_discrete(), "DQN only supports discrete action spaces"
        self.net = config.net("value")
        self.target_net = deepcopy(self.net)
        self.optimizer = config.optimizer(self.net.parameters())
        self.criterion = nn.MSELoss()
        self.policy = config.explorer()
        self.eval_policy = config.explorer(key="eval")
        self.replay = config.replay_buffer()
        assert self.replay.feed == DQNReplayFeed
        self.batch_indices = config.device.indices(config.replay_batch_size)

    def set_mode(self, train: bool = True) -> None:
        self.net.train(mode=train)

    def members_to_save(self) -> Tuple[str, ...]:
        return "net", "target_net", "policy", "total_steps"

    @torch.no_grad()
    def eval_action(self, state: Array) -> Action:
        return self.eval_policy.select_action(state, self.net).item()  # type: ignore

    def step(self, state: State) -> Tuple[State, float, bool, dict]:
        train_started = self.total_steps > self.config.train_start
        if train_started:
            action = self.policy.select_action(self.env.extract(state), self.net).item()
        else:
            action = self.env.spec.random_action()
        next_state, reward, done, info = self.env.step(action)
        self.replay.append(state, action, reward, next_state, done)
        if train_started:
            self._train()
        return next_state, reward, done, info

    @torch.no_grad()
    def _q_next(self, next_states: Array) -> Tensor:
        return self.target_net(next_states).max(1)[0]

    def _train(self) -> None:
        obs = self.replay.sample(self.config.replay_batch_size)
        obs = [ob.to_ndarray(self.env.extract) for ob in obs]
        states, actions, rewards, next_states, done = map(np.asarray, zip(*obs))
        q_next = self._q_next(next_states).mul_(self.tensor(1.0 - done))
        q_target = self.tensor(rewards).add_(q_next.mul_(self.config.discount_factor))
        q_current = self.net(states)[self.batch_indices, actions]
        loss = self.criterion(q_current, q_target)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        self.network_log(q_value=q_current.mean().item(), value_loss=loss.item())
        if (self.update_steps + 1) % self.config.sync_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())


class DoubleDQNAgent(DQNAgent):
    @torch.no_grad()
    def _q_next(self, next_states: Array) -> Tensor:
        """Returns Q values of next_states, supposing torch.no_grad() is called
        """
        q_next = self.target_net(next_states)
        q_values = self.net.q_values(next_states, nostack=True)
        return q_next[self.batch_indices, q_values.argmax(dim=-1)]
