from functools import partial
import numpy as np
from numpy import ndarray
import torch
from torch import nn, Tensor
from typing import Iterable, List, Tuple
from .a2c import A2cAgent
from ..config import Config
from ..envs import Action, State


class PPOAgent(A2cAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.penv = config.parallel_env()
        self.net = config.net('actor-critic')
        self.optimizer = config.optimizer(self.net.parameters())
        self.criterion = nn.MSELoss()

    def nstep(self, states: Iterable[State]) -> Tuple[Iterable[State], Iterable[float]]:
        rollouts, episodic_rewards = [], []
        for _ in range(self.config.nstep):
            states, rollout = self._one_step(states, episodic_rewards)
            rollouts.append(rollout)
        next_value = self.net(self.penv.states_to_array(states)).value.detach()
        rollouts.append((None, next_value, None, None))
        log_prob, entropy, value, return_, advantage = \
            map(partial(torch.cat, dim=0), zip(*self._calc_returns(next_value, rollouts)))
        policy_loss = -(log_prob * advantage).mean()
        value_loss = (return_ - value).pow(2).mean()
        entropy_loss = entropy.mean()
        self.optimizer.zero_grad()
        (policy_loss + self.config.value_loss_weight * value_loss -
         self.config.entropy_weight * entropy_loss).backward()
        nn.utils.clip_grad_norm_(self.net.parameters(), self.config.grad_clip)
        self.optimizer.step()
        return states, episodic_rewards
