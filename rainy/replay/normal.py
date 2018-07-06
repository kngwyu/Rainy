from numpy import ndarray
from .base import ReplayBuffer
from .array_deque import ArrayDeque


class NormalReplayBuffer(ReplayBuffer):
    def __init__(self, max_length: int = 1000):
        self.buf = ArrayDeque(default_size=max_length, fixed_len=True)
        self.cap = max_length

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
        self.buf.push(exp)
        if len(self) > self.cap:
            self.buf.popleft()

    def sample(self, batch_size: int):
        pass

