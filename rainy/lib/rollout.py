import numpy as np
import torch
from torch import Tensor
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
from typing import Generic, NamedTuple, Iterable, Iterator, List, Optional, Tuple
from ..envs import ParallelEnv, State
from ..net import Policy, recurrent, RnnBlock, RnnState
from ..utils import Device
from ..prelude import Array, GenericNamedMeta


class RolloutStorage(Generic[RnnState, State]):
    def __init__(self, nsteps: int, nworkers: int, device: Device) -> None:
        self.states: List[Array[State]] = []
        self.rewards: List[Array[float]] = []
        self.masks: List[Array[float]] = [np.ones(nworkers)]
        self.rnn_states: List[RnnState] = []
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
            rnn_state: Optional[RnnState] = None,
            policy: Optional[Policy] = None,
            value: Optional[Tensor] = None,
    ) -> None:
        assert self.states, '[RolloutStorage.push] Call set_initial_state first'
        self.states.append(state)
        self.rewards.append(reward)
        self.masks.append(1.0 - mask)
        if rnn_state is not None:
            self.rnn_states.append(rnn_state)
        if policy is not None:
            self.policies.append(policy)
        if value is not None:
            self.values.append(value.to(self.device.unwrapped))

    def reset(self) -> None:
        self.masks = [self.masks[-1]]
        self.states = [self.states[-1]]
        self.rnn_states = []
        self.rewards = []
        self.policies = []
        self.values = []

    def batch_states(self, penv: ParallelEnv) -> Tensor:
        states = [self.device.tensor(penv.states_to_array(s)) for s in self.states[:-1]]
        return torch.cat(states, dim=0)

    def batch_rnn_states(self, rnn: RnnBlock) -> RnnState:
        return rnn.make_batch(self.rnn_states, self.masks)

    def batch_actions(self) -> Tensor:
        return torch.cat([p.action() for p in self.policies], dim=0)

    def batch_returns(self) -> Tensor:
        return self.returns[:-1].flatten()

    def batch_values(self) -> Tensor:
        return torch.cat(self.values[:self.nsteps], dim=0)

    def batch_masks_and_rewards(self) -> Iterable[Tensor]:
        return map(lambda a: self.device.tensor(a).flatten(), (self.masks[:-1], self.rewards))

    def batch_log_probs(self) -> Tensor:
        return torch.cat([p.log_prob() for p in self.policies], dim=0)

    def _masks_and_rewards(self) -> Tuple[Tensor, Tensor]:
        return self.device.tensor(self.masks), self.device.tensor(self.rewards)

    def append_next_value(self, next_value: Tensor) -> None:
        self.returns[-1] = next_value
        self.values.append(next_value)

    def calc_ac_returns(self, next_value: Tensor, gamma: float) -> None:
        self.append_next_value(next_value)
        masks, rewards = self._masks_and_rewards()
        for i in reversed(range(self.nsteps)):
            self.returns[i] = self.returns[i + 1] * gamma * masks[i + 1] + rewards[i]

    def calc_gae_returns(self, next_value: Tensor, gamma: float, tau: float) -> None:
        self.append_next_value(next_value)
        masks, rewards = self._masks_and_rewards()
        gae = torch.zeros(self.nworkers, device=self.device.unwrapped)
        for i in reversed(range(self.nsteps)):
            td_error = \
                rewards[i] + gamma * self.values[i + 1] * masks[i + 1] - self.values[i]
            gae = td_error + gamma * tau * masks[i] * gae
            self.returns[i] = gae + self.values[i]


class RolloutBatch(NamedTuple, Generic[RnnState], metaclass=GenericNamedMeta):
    states: Tensor
    actions: Tensor
    masks: Tensor
    rewards: Tensor
    returns: Tensor
    values: Tensor
    old_log_probs: Tensor
    advantages: Tensor
    rnn_state: RnnState


class RolloutSampler(Generic[RnnState]):
    def __init__(
            self,
            storage: RolloutStorage,
            penv: ParallelEnv,
            minibatch_size: int,
            rnn: RnnBlock = recurrent.DummyRnn(),
            adv_normalize_eps: Optional[float] = None,
    ) -> None:
        """Create a batch sampler from storage for feed forward network.
        adv_normalize_eps is adpoted from Open AI's PPO implementation.
        I think it's for reduce variance of advantages, but I'm not sure.
        """
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
        self.masks, self.rewards = storage.batch_masks_and_rewards()
        self.returns = storage.batch_returns()
        self.values = storage.batch_values()
        self.old_log_probs = storage.batch_log_probs()
        self.rnn_states = storage.batch_rnn_states(rnn)
        self.advantages = self.returns - self.values
        if adv_normalize_eps:
            adv = self.advantages
            self.advantages = (adv - adv.mean()) / (adv.std() + adv_normalize_eps)

    def __iter__(self) -> Iterator[RolloutBatch]:
        samplar = BatchSampler(
            SubsetRandomSampler(range(self.batch_size)),
            batch_size=self.minibatch_size,
            drop_last=False
        )
        for indices in samplar:
            yield RolloutBatch(*map(lambda t: t[indices], (
                self.states,
                self.actions,
                self.masks,
                self.rewards,
                self.returns,
                self.values,
                self.old_log_probs,
                self.advantages,
                self.rnn_states
            )))

