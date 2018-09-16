from typing import Generic, List, Tuple
import numpy as np
from numpy import ndarray
from .base import ReplayBuffer, State
from .array_deque import ArrayDeque

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
        return [self.buf[idx] for idx in np.random.choice(n, batch_size)]

    def __len__(self):
        return len(self.buf)


