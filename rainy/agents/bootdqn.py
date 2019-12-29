"""
This module has an implementation of Bootstrapped DQN, which is described in
- Deep Exploration via Bootstrapped DQN
  - URL: https://arxiv.org/abs/1602.04621
- Randomized Prior Functions for Deep Reinforcement Learning
  - URL: https://arxiv.org/abs/1806.03335
"""
import copy
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Tuple
from .base import DQNLikeAgent
from ..config import Config
from ..prelude import Action, Array, State
from ..replay import BootDQNReplayFeed


class BootDQNAgent(DQNLikeAgent):
    SAVED_MEMBERS = "net", "policy", "total_steps", "target_net"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if not self.env.spec.is_discrete():
            raise RuntimeError("DQN only supports discrete action space.")
        self.net = config.net("bootdqn")
        self.target_net = copy.deepcopy(self.net)
        self.optimizer = config.optimizer(self.net.parameters())
        self.policy = config.explorer()
        self.eval_policy = config.explorer(key="eval")
        self.replay = config.replay_buffer()
        self.active_head = 0
        if self.replay.feed is not BootDQNReplayFeed:
            raise RuntimeError("BootDQNAgent needs BootDQNReplayFeed")

    def set_mode(self, train: bool = True) -> None:
        self.net.train(mode=train)

    @torch.no_grad()
    def eval_action(self, state: Array) -> Action:
        return self.eval_policy.select_action(state, self.net).item()  # type: ignore

    def on_terminal(self) -> None:
        self.active_head = np.random.randint(self.config.num_ensembles)

    def action(self, state: State) -> Action:
        if self.train_started:
            with torch.no_grad():
                qs = self.net.q_i_s(self.active_head, self.env.extract(state)).detach()
            return self.policy.select_from_value(qs).item()
        else:
            return self.env.spec.random_action()

    def step(self, state: State) -> Tuple[State, float, bool, dict]:
        action = self.action(state)
        next_state, reward, done, info = self.env.step(action)
        randn = np.random.uniform(0, 1, self.config.num_ensembles)
        mask = randn < self.config.replay_prob
        self.replay.append(state, action, reward, next_state, done, mask)
        return next_state, reward, done, info

    @torch.no_grad()
    def _q_next(self, next_states: Array) -> Tensor:
        return self.net(next_states).max(axis=-1)[0]

    def train(self, replay_feed: BootDQNReplayFeed) -> None:
        gamma = self.config.discount_factor
        obs = [ob.to_array(self.env.extract) for ob in replay_feed]
        states, actions, rewards, next_states, done, mask = map(np.asarray, zip(*obs))
        q_next = self._q_next(next_states)
        r = self.tensor(rewards).view(-1, 1)
        q_target = r + q_next * self.tensor(1.0 - done).mul_(gamma).view(-1, 1)
        q_current = self.net.q_s_a(states, actions)
        loss = F.mse_loss(q_current, q_target, reduction="none")
        masked_loss = loss.masked_select(self.tensor(mask, dtype=torch.bool)).mean()
        self._backward(masked_loss, self.optimizer, self.net.parameters())
        self.network_log(q_value=q_current.mean().item(), value_loss=masked_loss.item())
        if (self.update_steps + 1) % self.config.sync_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())
