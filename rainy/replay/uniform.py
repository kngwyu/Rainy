from numpy import ndarray
from typing import Callable, Generic, List, NamedTuple, Tuple
from .array_deque import ArrayDeque
from .base import ReplayFeed, ReplayBuffer
from ..envs.ext import State
from ..utils import sample_indices
from ..prelude import GenericNamedMeta


class UniformReplayBuffer(ReplayBuffer, Generic[ReplayFeed]):
    def __init__(self, feed: ReplayFeed, capacity: int = 1000) -> None:
        super().__init__(feed)
        self.buf = ArrayDeque(capacity=capacity)
        self.cap = capacity

    def append(self, *args) -> None:
        exp = self.feed(*args)
        self.buf.push_back(exp)
        if len(self) > self.cap:
            self.buf.pop_front()

    def sample(self, batch_size: int) -> List[ReplayFeed]:
        return [self.buf[idx] for idx in sample_indices(len(self.buf), batch_size)]

    def __len__(self):
        return len(self.buf)


class DqnReplayFeed(NamedTuple, Generic[State], metaclass=GenericNamedMeta):
    state: State
    action: int
    reward: float
    next_state: State
    done: bool

    def to_ndarray(
            self,
            wrap: Callable[[State], ndarray]
    ) -> Tuple[ndarray, int, float, ndarray, bool]:
        return (
            wrap(self.state),
            self.action,
            self.reward,
            wrap(self.next_state),
            self.done,
        )

