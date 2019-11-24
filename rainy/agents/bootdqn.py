"""
This module has an implementation of Bootstrapped DQN, which is described in
- Deep Exploration via Bootstrapped DQN(https://arxiv.org/abs/1602.04621)
- Randomized Prior Functions for Deep Reinforcement Learning(https://arxiv.org/abs/1806.03335)
"""
import copy
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Tuple
from .base import OneStepAgent
from ..config import Config
from ..prelude import Action, Array, State
from ..replay import BootDQNReplayFeed


class EpisodicBootDQNAgent(OneStepAgent):
    SAVED_MEMBERS = "net", "policy", "total_steps"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if not self.env.spec.is_discrete():
            raise RuntimeError("DQN only supports discrete action space.")
        self.net = config.net("bootdqn")
        self.optimizer = config.optimizer(self.net.parameters())
        self.policy = config.explorer()
        self.eval_policy = config.explorer(key="eval")
        self.replay = config.replay_buffer()
        self.replay.allow_overlap = True
        self.active_head = 0
        if self.replay.feed is not BootDQNReplayFeed:
            raise RuntimeError("BootDQNAgent needs BootDQNReplayFeed")

    def set_mode(self, train: bool = True) -> None:
        self.net.train(mode=train)

    @torch.no_grad()
    def eval_action(self, state: Array) -> Action:
        return self.eval_policy.select_action(state, self.net).item()  # type: ignore

    def step(self, state: State) -> Tuple[State, float, bool, dict]:
        with torch.no_grad():
            qs = self.net.q_i_s(self.active_head, self.env.extract(state)).detach()
        action = self.policy.select_from_value(qs).item()
        next_state, reward, done, info = self.env.step(action)
        self._append_to_replay(state, action, reward, next_state, done)
        if done:
            self._train()
            self.active_head = np.random.randint(self.config.num_ensembles)
        return next_state, reward, done, info

    def _append_to_replay(self, *transition) -> None:
        n_ens = self.config.num_ensembles
        mask = np.random.uniform(0, 1, n_ens) < self.config.replay_prob
        self.replay.append(*transition, mask)

    @torch.no_grad()
    def _q_next(self, next_states: Array) -> Tensor:
        return self.net(next_states).max(axis=-1)[0]

    def _train(self) -> None:
        gamma = self.config.discount_factor
        obs = self.replay.sample(self.config.replay_batch_size)
        obs = [ob.to_array(self.env.extract) for ob in obs]
        states, actions, rewards, next_states, done, mask = map(np.asarray, zip(*obs))
        q_next = self._q_next(next_states)
        r = self.tensor(rewards).view(-1, 1)
        q_target = r + q_next * self.tensor(1.0 - done).mul_(gamma).view(-1, 1)
        q_current = self.net.q_s_a(states, actions)
        loss = F.mse_loss(q_current, q_target, reduction="none")
        masked_loss = loss.masked_select(self.tensor(mask, dtype=torch.bool)).mean()
        self._backward(masked_loss, self.optimizer, self.net.parameters())
        self.network_log(q_value=q_current.mean().item(), value_loss=masked_loss.item())


class BootDQNAgent(EpisodicBootDQNAgent):
    SAVED_MEMBERS = "net", "policy", "total_steps", "target_net"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.target_net = copy.deepcopy(self.net)
        self.replay.allow_overlap = False

    def step(self, state: State) -> Tuple[State, float, bool, dict]:
        train_started = self.total_steps > self.config.train_start
        if train_started:
            with torch.no_grad():
                qs = self.net.q_i_s(self.active_head, self.env.extract(state)).detach()
            action = self.policy.select_from_value(qs).item()
        else:
            action = self.env.spec.random_action()
        next_state, reward, done, info = self.env.step(action)
        self._append_to_replay(state, action, reward, next_state, done)
        if done:
            self.active_head = np.random.randint(self.config.num_ensembles)
        if train_started:
            self._train()
        return next_state, reward, done, info

    @torch.no_grad()
    def _q_next(self, next_states: Array) -> Tensor:
        return self.net(next_states).max(axis=-1)[0]

    def _train(self) -> None:
        super()._train
        if (self.update_steps + 1) % self.config.sync_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())
