from typing import Callable, Generic, List, NamedTuple, Tuple, Type

import numpy as np

from ..prelude import Array, GenericNamedMeta, State
from ..utils import sample_indices
from .array_deque import ArrayDeque
from .base import ReplayBuffer, ReplayFeed


class UniformReplayBuffer(ReplayBuffer, Generic[ReplayFeed]):
    def __init__(
        self, feed: Type[ReplayFeed], capacity: int = 1000, allow_overlap: bool = False
    ) -> None:
        super().__init__(feed, allow_overlap=allow_overlap)
        self.buf = ArrayDeque(capacity=capacity)
        self.cap = capacity

    def append(self, *args) -> None:
        exp = self.feed(*args)
        self.buf.push_back(exp)
        if len(self) > self.cap:
            self.buf.pop_front()

    def sample(self, batch_size: int) -> List[ReplayFeed]:
        if self.allow_overlap:
            indices = np.random.randint(len(self.buf), size=batch_size)
        else:
            indices = sample_indices(len(self.buf), batch_size)
        return [self.buf[idx] for idx in indices]

    def __len__(self):
        return len(self.buf)


class DQNReplayFeed(NamedTuple, Generic[State], metaclass=GenericNamedMeta):
    state: State
    action: int
    next_state: State
    reward: float
    done: bool

    def to_array(
        self, wrap: Callable[[State], Array]
    ) -> Tuple[Array[float], int, Array[float], float, bool]:
        return (
            wrap(self.state),
            self.action,
            wrap(self.next_state),
            self.reward,
            self.done,
        )


class BootDQNReplayFeed(NamedTuple, Generic[State], metaclass=GenericNamedMeta):
    state: State
    action: int
    next_state: State
    reward: float
    done: bool
    ensemble_mask: Array[bool]

    def to_array(
        self, wrap: Callable[[State], Array]
    ) -> Tuple[Array[float], int, Array[float], float, bool, Array[bool]]:
        return (
            wrap(self.state),
            self.action,
            wrap(self.next_state),
            self.reward,
            self.done,
            self.ensemble_mask,
        )
