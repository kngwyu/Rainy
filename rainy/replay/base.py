from abc import ABC, abstractmethod
from numpy import ndarray

class ReplayBuffer(ABC):
    @abstractmethod
    def append(
            self,
            state: ndarray,
            action: int,
            reward: float,
            next_state: ndarray,
            is_terminal: bool,
    ) -> None:
        pass

    @abstractmethod
    def sample(self, n: int):
        pass

    @abstractmethod
    def __len__(self):
        pass

    @abstractmethod
    def save(self, filename):
        pass

    @abstractmethod
    def load(self, filename):
        pass

