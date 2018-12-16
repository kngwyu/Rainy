import numpy as np
import torch
from torch import Tensor
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from typing import Any, Generic, Iterable, List, NamedTuple, Optional, Tuple
from ..envs import ParallelEnv, State
from ..net import Policy
from ..util import Device
from ..util.meta import Array


class RolloutStorage(Generic[State]):
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
        self.masks.append(1.0 - mask)  # type: ignore
        if policy is not None:
            self.policies.append(policy)
        if value is not None:
            self.values.append(value.to(self.device.unwrapped))  # for testing

    def reset(self) -> None:
        self.masks = [self.masks[-1]]
        self.states = [self.states[-1]]
        self.rewards = []
        self.policies = []
        self.values = []

    def batch_states(self, penv: ParallelEnv) -> Tensor:
        states = [self.device.tensor(penv.states_to_array(s)) for s in self.states[:-1]]
        return torch.cat(states, dim=0)

    def batch_actions(self) -> Tensor:
        return torch.cat([p.action() for p in self.policies], dim=0)

    def batch_returns(self) -> Tensor:
        return self.returns[:-1].flatten()

    def _values(self) -> Tensor:
        return torch.cat(self.values[:self.nsteps], dim=0)

    def _masks_and_rewards(self) -> Tuple[Tensor, Tensor]:
        m, r = self.device.tensor(self.masks[:-1]), self.device.tensor(self.rewards)
        return m.flatten(), r.flatten()

    def calc_ac_returns(self, next_value: Tensor, gamma: float) -> None:
        self.returns[-1] = next_value
        masks, rewards = self._masks_and_rewards(next_value)
        for i in reversed(range(self.nsteps)):
            self.returns[i] = self.returns[i + 1] * gamma * masks[i + 1] + rewards[i]

    def calc_gae_returns(self, next_value: Tensor, gamma: float, tau: float) -> None:
        self.values.append(next_value)
        masks, rewards = self._masks_and_rewards(next_value)
        gae = torch.zeros(self.nworkers, device=self.device.unwrapped)
        for i in reversed(range(self.nsteps)):
            td_error = \
                rewards[i] + gamma * self.values[i + 1] * masks[i + 1] - self.values[i]
            gae = td_error + gamma * tau * masks[i] * gae
            self.returns[i] = gae + self.values[i]

    def _log_probs(self) -> Tensor:
        return torch.cat([p.log_prob() for p in self.policies], dim=0)


class FeedForwardBatch(NamedTuple):
    states: Tensor
    actions: Tensor
    masks: Tensor
    rewards: Tensor
    returns: Tensor
    values: Tensor
    old_log_probs: Tensor
    advantages: Tensor


class FeedForwardSampler:
    def __init__(
            self,
            storage: RolloutStorage[Any],
            penv: ParallelEnv,
            minibatch_size: int,
            adv_normalize_eps: Optional[float] = None,
    ) -> None:
        self.batch_size = storage.nsteps * storage.nworkers
        if minibatch_size >= self.batch_size:
            raise ValueError(
                'PPO requires minibatch_size <= nsteps * nworkers, but '
                'minibatch_size: {}, nsteps: {}, nworkers: {} was passed.'
                .format(minibatch_size, storage.nsteps, storage.nworkers)
            )
        self.minibatch_size = minibatch_size
        self.states = storage.batch_states(penv)
        self.actions = storage.batch_actions()
        self.masks, self.rewards = storage._masks_and_rewards()
        self.returns = storage.batch_returns()
        self.values = storage._values()
        self.old_log_probs = storage._log_probs()
        print(self.returns)
        print(self.values)
        self.advantages = self.returns - self.values
        if adv_normalize_eps:
            # I don't know what this is for, but baselines does so ('_')
            adv = self.advantages
            self.advantages = (adv - adv.mean()) / (adv.std() + adv_normalize_eps)

    def __iter__(self) -> Iterable[tuple]:
        samplar = BatchSampler(
            SubsetRandomSampler(range(self.batch_size)),
            batch_size=self.minibatch_size,
            drop_last=False
        )
        for indices in samplar:
            yield FeedForwardBatch(*map(lambda t: t[indices], (
                self.states,
                self.actions,
                self.masks,
                self.rewards,
                self.returns,
                self.values,
                self.old_log_probs,
                self.advantages
            )))

