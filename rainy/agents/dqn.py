"""
This module has an implementation of DQN and Double DQN.
Corresponding papers:
- Human-level control through deep reinforcement learning
  - https://www.nature.com/articles/nature14236/
- Deep Reinforcement Learning with Double Q-Learning
  - https://www.aaai.org/ocs/index.php/AAAI/AAAI16/paper/viewPaper/12389
"""

from copy import deepcopy
from typing import Optional

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from ..config import Config
from ..envs import ParallelEnv
from ..prelude import Action, Array, State
from ..replay import DQNReplayFeed
from .base import DQNLikeAgent, Netout


class DQNAgent(DQNLikeAgent):
    SAVED_MEMBERS = "net", "policy", "total_steps", "target_net"
    SUPPORT_PARALLEL_ENV = True

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if not self.env._spec.is_discrete():
            raise RuntimeError("DQN only supports discrete action space.")
        self.net = config.net("dqn")
        self.target_net = deepcopy(self.net)
        self.optimizer = config.optimizer(self.net.parameters())
        self.policy = config.explorer()
        self.eval_policy = config.explorer(key="eval")
        self.replay = config.replay_buffer()
        if self.replay.feed is not DQNReplayFeed:
            raise RuntimeError("DQNAgent needs DQNReplayFeed")
        self.batch_indices = config.device.indices(config.replay_batch_size)

    def set_mode(self, train: bool = True) -> None:
        self.net.train(mode=train)

    @torch.no_grad()
    def eval_action(self, state: Array, net_outputs: Optional[Netout] = None) -> Action:
        return self.eval_policy.select_action(state, self.net, net_outputs).item()

    def action(self, state: State) -> Action:
        if self.train_started:
            return self.policy.select_action(self.env.extract(state), self.net).item()
        else:
            return self.env._spec.random_action()

    def batch_actions(self, states: Array[State], penv: ParallelEnv) -> Array[Action]:
        if self.train_started:
            states = penv.extract(states)
            return self.policy.select_action(states, self.net).squeeze_().numpy()
        else:
            return self.env._spec.random_actions(states.shape[0])

    @torch.no_grad()
    def _q_next(self, next_states: Array) -> Tensor:
        return self.target_net(next_states).max(axis=-1)[0]

    def train(self, replay_feed: DQNReplayFeed) -> None:
        obs = [ob.to_array(self.env.extract) for ob in replay_feed]
        states, actions, next_states, rewards, done = map(np.asarray, zip(*obs))
        q_next = self._q_next(next_states).squeeze_()
        discount = self.tensor(1.0 - done).mul_(self.config.discount_factor)
        q_target = self.tensor(rewards).add_(q_next.mul_(discount))
        q_prediction = self.net(states)[self.batch_indices, actions]
        loss = F.mse_loss(q_prediction, q_target)
        self._backward(loss, self.optimizer, self.net.parameters())
        self.network_log(q_value=q_prediction.mean().item(), value_loss=loss.item())
        if self.update_steps > 0 and self.update_steps % self.config.sync_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())


class DoubleDQNAgent(DQNAgent):
    @torch.no_grad()
    def _q_next(self, next_states: Array) -> Tensor:
        """Returns Q values of next_states, supposing torch.no_grad() is called
        """
        q_next = self.target_net(next_states)
        q_value = self.net.q_value(next_states, nostack=True)
        return q_next[self.batch_indices, q_value.argmax(dim=-1)]
