import torch
from torch import Tensor
from typing import Generic, NamedTuple, Iterator, List, Optional, Tuple
from ..envs import ParallelEnv, State
from ..net import DummyRnn, Policy, RnnBlock, RnnState
from ..utils import Device
from ..utils.sample import FeedForwardBatchSampler, RecurrentBatchSampler
from ..prelude import Array


class RolloutStorage(Generic[State]):
    def __init__(self, nsteps: int, nworkers: int, device: Device) -> None:
        self.states: List[Array[State]] = []
        self.rewards: List[Array[float]] = []
        self.masks: List[Tensor] = [device.zeros(nworkers)]
        self.rnn_states: List[RnnState] = []
        self.policies: List[Policy] = []
        self.values: List[Tensor] = []
        self.returns: Tensor = device.zeros((nsteps + 1, nworkers))
        self.nsteps = nsteps
        self.nworkers = nworkers
        self.device = device

    def initialized(self) -> bool:
        return len(self.states) != 0

    def set_initial_state(
            self,
            state: Array[State],
            rnn_state: RnnState = DummyRnn.DUMMY_STATE
    ) -> None:
        self.states.append(state)
        self.rnn_states.append(rnn_state)

    def push(
            self,
            state: Array[State],
            reward: Array[float],
            mask: Array[bool],
            rnn_state: RnnState = DummyRnn.DUMMY_STATE,
            policy: Optional[Policy] = None,
            value: Optional[Tensor] = None,
    ) -> None:
        assert self.states, '[RolloutStorage.push] Call set_initial_state first'
        self.states.append(state)
        self.rewards.append(reward)
        self.masks.append(self.device.tensor(1.0 - mask))
        self.rnn_states.append(rnn_state)
        if policy is not None:
            self.policies.append(policy)
        if value is not None:
            self.values.append(value.to(self.device.unwrapped))

    def reset(self) -> None:
        self.masks = [self.masks[-1]]
        self.states = [self.states[-1]]
        self.rnn_states = [self.rnn_states[-1]]
        self.rewards = []
        self.policies = []
        self.values = []

    def batch_states(self, penv: ParallelEnv) -> Tensor:
        states = [self.device.tensor(penv.extract(s)) for s in self.states[:-1]]
        return torch.cat(states, dim=0)

    def batch_actions(self) -> Tensor:
        return torch.cat([p.action() for p in self.policies])

    def batch_returns(self) -> Tensor:
        return self.returns[:-1].flatten()

    def batch_values(self) -> Tensor:
        return torch.cat(self.values[:self.nsteps])

    def batch_masks(self) -> Tensor:
        return torch.cat(self.masks[:-1])

    def batch_rewards(self) -> Tensor:
        return self.device.tensor(self.rewards).flatten()

    def batch_log_probs(self) -> Tensor:
        return torch.cat([p.log_prob() for p in self.policies])

    def _calc_ret_common(self, next_value: Tensor) -> Tuple[Tensor, Tensor]:
        self.returns[-1] = next_value
        self.values.append(next_value)
        return torch.stack(self.masks), self.device.tensor(self.rewards)

    def calc_ac_returns(self, next_value: Tensor, gamma: float) -> None:
        masks, rewards = self._calc_ret_common(next_value)
        for i in reversed(range(self.nsteps)):
            self.returns[i] = self.returns[i + 1] * gamma * masks[i + 1] + rewards[i]

    def calc_gae_returns(self, next_value: Tensor, gamma: float, tau: float) -> None:
        masks, rewards = self._calc_ret_common(next_value)
        gae = self.device.zeros(self.nworkers)
        for i in reversed(range(self.nsteps)):
            td_error = \
                rewards[i] + gamma * self.values[i + 1] * masks[i + 1] - self.values[i]
            gae = td_error + gamma * tau * masks[i] * gae
            self.returns[i] = gae + self.values[i]


class RolloutBatch(NamedTuple):
    states: Tensor
    actions: Tensor
    masks: Tensor
    rewards: Tensor
    returns: Tensor
    values: Tensor
    old_log_probs: Tensor
    advantages: Tensor
    rnn_init: RnnState


class RolloutSampler:
    def __init__(
            self,
            storage: RolloutStorage,
            penv: ParallelEnv,
            minibatch_size: int,
            rnn: RnnBlock = DummyRnn(),
            adv_normalize_eps: Optional[float] = None,
    ) -> None:
        """Create a batch sampler from storage for feed forward network.
        adv_normalize_eps is adpoted from Open AI's PPO implementation.
        I think it's for reduce variance of advantages, but I'm not sure.
        """
        self.nworkers = storage.nworkers
        self.nsteps = storage.nsteps
        if minibatch_size >= self.nworkers * self.nsteps:
            raise ValueError(
                'PPO requires minibatch_size <= nsteps * nworkers, but '
                'minibatch_size: {}, nsteps: {}, nworkers: {} was passed.'
                .format(minibatch_size, storage.nsteps, storage.nworkers)
            )
        self.minibatch_size = minibatch_size
        self.states = storage.batch_states(penv)
        self.actions = storage.batch_actions()
        self.masks = storage.batch_masks()
        self.returns = storage.batch_returns()
        self.rewards = storage.batch_rewards()
        self.values = storage.batch_values()
        self.old_log_probs = storage.batch_log_probs()
        self.rnn_init = storage.rnn_states[0]
        self.advantages = self.returns - self.values
        if adv_normalize_eps:
            adv = self.advantages
            self.advantages = (adv - adv.mean()) / (adv.std() + adv_normalize_eps)

    def __iter__(self) -> Iterator[RolloutBatch]:
        sampler_cls = FeedForwardBatchSampler \
            if self.rnn_init is DummyRnn.DUMMY_STATE else RecurrentBatchSampler
        for i in sampler_cls(self.nsteps, self.nworkers, self.minibatch_size):  # type: ignore
            yield RolloutBatch(
                self.states[i],
                self.actions[i],
                self.masks[i],
                self.rewards[i],
                self.returns[i],
                self.values[i],
                self.old_log_probs[i],
                self.advantages[i],
                self.rnn_init[i[:len(i) // self.nsteps]]
            )

