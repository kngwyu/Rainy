from copy import deepcopy
import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F
from typing import Tuple
from .base import OneStepAgent
from ..config import Config
from ..prelude import Action, Array, State
from ..replay import DQNReplayFeed


class EpisodicDQNAgent(OneStepAgent):
    """A DQN variant, which has no target network and
    updates the target only once per episode.
    """

    SAVED_MEMBERS = "net", "policy", "total_steps"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if not self.env.spec.is_discrete():
            raise RuntimeError("DQN only supports discrete action space.")
        self.net = config.net("dqn")
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
    def eval_action(self, state: Array) -> Action:
        return self.eval_policy.select_action(state, self.net).item()  # type: ignore

    def step(self, state: State) -> Tuple[State, float, bool, dict]:
        action = self.policy.select_action(self.env.extract(state), self.net).item()
        next_state, reward, done, info = self.env.step(action)
        self.replay.append(state, action, reward, next_state, done)
        if done:
            self._train()
        return next_state, reward, done, info

    @torch.no_grad()
    def _q_next(self, next_states: Array) -> Tensor:
        return self.net(next_states).max(axis=-1)[0]

    def _train(self) -> None:
        obs = self.replay.sample(self.config.replay_batch_size)
        obs = [ob.to_array(self.env.extract) for ob in obs]
        states, actions, rewards, next_states, done = map(np.asarray, zip(*obs))
        q_next = self._q_next(next_states).mul_(self.tensor(1.0 - done))
        q_target = self.tensor(rewards).add_(q_next.mul_(self.config.discount_factor))
        q_current = self.net(states)[self.batch_indices, actions]
        loss = F.mse_loss(q_current, q_target)
        self._backward(loss, self.optimizer, self.net.parameters())
        self.network_log(q_value=q_current.mean().item(), value_loss=loss.item())


class DQNAgent(EpisodicDQNAgent):

    SAVED_MEMBERS = "net", "policy", "total_steps", "target_net"

    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.target_net = deepcopy(self.net)

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
        return self.target_net(next_states).max(axis=-1)[0]

    def _train(self):
        super()._train()
        if (self.update_steps + 1) % self.config.sync_freq == 0:
            self.target_net.load_state_dict(self.net.state_dict())


class DoubleDQNAgent(DQNAgent):
    @torch.no_grad()
    def _q_next(self, next_states: Array) -> Tensor:
        """Returns Q values of next_states, supposing torch.no_grad() is called
        """
        q_next = self.target_net(next_states)
        q_value = self.net.q_value(next_states, nostack=True)
        return q_next[self.batch_indices, q_value.argmax(dim=-1)]
