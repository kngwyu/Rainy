import numpy as np
import torch
from torch import Tensor
from typing import List, Optional, Tuple
from ..envs import ParallelEnv, State
from ..net import Policy
from ..util import Device
from ..util.meta import Array


class RolloutStorage:
    def __init__(self, nsteps: int, nworkers: int, device: Device) -> None:
        self.states: List[Array[State]] = []
        self.rewards: List[Array[float]] = []
        self.masks: List[Array[float]] = [np.ones(nworkers)]
        self.policies: List[Policy] = []
        self.values: List[Tensor] = []
        self.returns: Tensor = torch.zeros(nsteps + 1, nworkers, device=device.unwrapped)
        self.nsteps = nsteps
        self.nworkers = nworkers
        self.device = device

    def initialized(self) -> bool:
        return len(self.states) != 0

    def set_initial_state(self, state: Array[State]) -> None:
        self.states.append(state)

    def push(
            self,
            state: Array[State],
            reward: Array[float],
            mask: Array[bool],
            policy: Optional[Policy] = None,
            value: Optional[Tensor] = None,
    ) -> None:
        assert self.states, '[RolloutStorage.push] Call set_initial_state first'
        self.states.append(state)
        self.rewards.append(reward)
        self.masks.append(1.0 - mask)
        if policy is not None:
            self.policies.append(policy)
        if value is not None:
            self.values.append(value)

    def reset(self) -> None:
        self.masks = [self.masks[-1]]
        self.states = [self.states[-1]]
        self.rewards = []
        self.policies = []
        self.values = []

    def batched_states(self, penv: ParallelEnv) -> Tensor:
        states = [self.device.tensor(penv.states_to_array(s)) for s in self.states[:-1]]
        return torch.cat(states, dim=0)

    def batched_actions(self) -> Tensor:
        return torch.cat([p.action() for p in self.policies], dim=0)

    def batched_returns(self) -> Tensor:
        return self.returns[:-1].flatten()

    def _masks_and_rewards(self, next_value: Tensor) -> Tuple[Tensor, Tensor]:
        self.returns[-1] = next_value
        self.values.append(next_value)
        return self.device.tensor(self.masks), self.device.tensor(self.rewards)

    def calc_ac_returns(self, next_value: Tensor, gamma: float) -> None:
        masks, rewards = self._masks_and_rewards(next_value)
        for i in reversed(range(self.nsteps)):
            self.returns[i] = self.returns[i + 1] * gamma * masks[i + 1] + rewards[i]

    def calc_gae_returns(self, next_value: Tensor, gamma: float, tau: float) -> None:
        masks, rewards = self._masks_and_rewards(next_value)
        gae = torch.zeros(self.nworkers, device=self.device.unwrapped)
        for i in reversed(range(self.nsteps)):
            td_error = rewards[i] + gamma * self.values[i + 1] * masks[i + 1] - self.values[i]
            gae = td_error + gamma * tau * masks[i] * gae
            self.returns[i] = gae + self.values[i]

