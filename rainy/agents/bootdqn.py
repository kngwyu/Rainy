"""
This module has an implementation of Bootstrapped DQN, which is described in
- Deep Exploration via Bootstrapped DQN
  - URL: https://arxiv.org/abs/1602.04621
- Randomized Prior Functions for Deep Reinforcement Learning
  - URL: https://arxiv.org/abs/1806.03335
"""
import copy
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ..config import Config
from ..envs import EnvTransition
from ..prelude import Action, Array, State
from ..replay import BootDQNReplayFeed
from .base import DQNLikeAgent, Netout


class BootDQNAgent(DQNLikeAgent):
    SAVED_MEMBERS = "net", "policy", "total_steps", "target_net"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if not self.env._spec.is_discrete():
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
    def eval_action(self, state: Array, net_outputs: Optional[Netout] = None) -> Action:
        return self.eval_policy.select_action(state, self.net, net_outputs).item()

    def action(self, state: State) -> Action:
        if self.train_started:
            with torch.no_grad():
                qs = self.net.q_i_s(self.active_head, self.env.extract(state)).detach()
            return self.policy.select_from_value(qs).item()
        else:
            return self.env._spec.random_action()

    def store_transition(
        self,
        state: State,
        action: Action,
        next_state: State,
        reward: float,
        terminal: bool,
    ) -> None:
        randn = np.random.uniform(0, 1, self.config.num_ensembles)
        # Masks some observations per ensemble
        mask = randn < self.config.replay_prob
        self.replay.append(state, action, next_state, reward, terminal, mask)
        # If the episode ends, change the executing policy
        if terminal:
            self.active_head = np.random.randint(self.config.num_ensembles)

    @torch.no_grad()
    def _q_next(self, next_states: Array) -> Tensor:
        return self.net(next_states).max(axis=-1)[0]

    def train(self, replay_feed: BootDQNReplayFeed) -> None:
        obs = [ob.to_array(self.env.extract) for ob in replay_feed]
        states, actions, next_states, rewards, done, mask = map(np.asarray, zip(*obs))
        q_next = self._q_next(next_states)
        r = self.tensor(rewards).unsqueeze_(-1)
        discount = self.tensor(1.0 - done).mul_(self.config.discount_factor)
        q_target = r + discount.unsqueeze_(-1) * q_next
        q_current = self.net.q_s_a(states, actions)
        loss = F.mse_loss(q_current, q_target, reduction="none")
        masked_loss = loss.masked_select(self.tensor(mask, dtype=torch.bool)).mean()
        self._backward(masked_loss, self.optimizer, self.net.parameters())
        self.network_log(q_value=q_current.mean().item(), value_loss=masked_loss.item())
        if self.update_steps > 0 and self.update_steps % self.config.sync_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())
