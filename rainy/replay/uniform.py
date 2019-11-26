import numpy as np
from typing import Callable, Generic, List, NamedTuple, Tuple, Type
from .array_deque import ArrayDeque
from .base import ReplayFeed, ReplayBuffer
from ..utils import sample_indices
from ..prelude import Array, GenericNamedMeta, State


class UniformReplayBuffer(ReplayBuffer, Generic[ReplayFeed]):
    def __init__(self, feed: Type[ReplayFeed], capacity: int = 1000) -> None:
        super().__init__(feed)
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
    reward: float
    next_state: State
    done: bool

    def to_array(
        self, wrap: Callable[[State], Array]
    ) -> Tuple[Array[float], int, float, Array[float], bool]:
        return (
            wrap(self.state),
            self.action,
            self.reward,
            wrap(self.next_state),
            self.done,
        )


class BootDQNReplayFeed(NamedTuple, Generic[State], metaclass=GenericNamedMeta):
    state: State
    action: int
    reward: float
    next_state: State
    done: bool
    ensemble_mask: Array[bool]

    def to_array(
        self, wrap: Callable[[State], Array]
    ) -> Tuple[Array[float], int, float, Array[float], bool, Array[bool]]:
        return (
            wrap(self.state),
            self.action,
            self.reward,
            wrap(self.next_state),
            self.done,
            self.ensemble_mask,
        )
