from numpy import ndarray
from typing import Callable, Generic, List, Tuple
from .array_deque import ArrayDeque
from .base import ReplayBuffer, State
from ..util import sample_indices


class UniformReplayBuffer(ReplayBuffer, Generic[State]):
    def __init__(self, capacity: int = 1000) -> None:
        self.buf = ArrayDeque(capacity=capacity)
        self.cap = capacity

    def append(
            self,
            state: State,
            action: int,
            reward: float,
            next_state: State,
            is_terminal: bool,
    ) -> None:
        exp = (state, action, reward, next_state, is_terminal)
        self.buf.push_back(exp)
        if len(self) > self.cap:
            self.buf.pop_front()

    def sample(self, batch_size: int) -> List[Tuple[State, int, float, State, bool]]:
        n = len(self.buf)
        return [self.buf[idx] for idx in sample_indices(n, batch_size)]

    def sample_with_state_wrapper(
            self,
            batch_size: int,
            wrap_state: Callable[[State], ndarray],
    ) -> List[Tuple[ndarray, int, float, ndarray, bool]]:
        n = len(self.buf)
        return [(
            wrap_state(self.buf[idx][0]),
            self.buf[idx][1],
            self.buf[idx][2],
            wrap_state(self.buf[idx][3]),
            self.buf[idx][4])
            for idx in sample_indices(n, batch_size)]

    def __len__(self):
        return len(self.buf)


