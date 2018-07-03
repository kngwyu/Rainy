from collections import deque
from numpy import ndarray
from .base import ReplayBuffer

class NormalReplayBuffer(ReplayBuffer):
    def __init__(self, max_length: int = 1000):
        self.buf = deque()
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
        self.buf.append(exp)
        if len(self) > self.cap:
            self.buf.popleft()

    def sample(self, batch_size: int):
        pass

