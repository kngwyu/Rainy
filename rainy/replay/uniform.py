import numpy as np
from numpy import ndarray
from .base import ReplayBuffer
from .array_deque import ArrayDeque


class UniformReplayBuffer(ReplayBuffer):
    def __init__(self, capacity: int = 1000) -> None:
        self.buf = ArrayDeque(capacity=capacity)
        self.cap = capacity

    def append(
            self,
            state: ndarray,
            action: int,
            reward: float,
            next_state: ndarray,
            is_terminal: bool,
    ) -> None:
        exp = dict(
            state=state,
            action=action,
            reward=reward,
            next_state=next_state,
            is_terminal=is_terminal
        )
        self.buf.push_back(exp)
        if len(self) > self.cap:
            self.buf.pop_front()

    def sample(self, batch_size: int):
        n = len(self.buf)
        return [self.buf[idx] for idx in np.random.choice(n, batch_size)]

    def __len__(self):
        return len(self.buf)


