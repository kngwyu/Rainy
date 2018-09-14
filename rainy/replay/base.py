from abc import ABC, abstractmethod
from typing import Generic, List, Tuple, TypeVar

State = TypeVar('State')

class ReplayBuffer(ABC, Generic[State]):
    @abstractmethod
    def append(
            self,
            state: State,
            action: int,
            reward: float,
            next_state: State,
            is_terminal: bool,
    ) -> None:
        pass

    @abstractmethod
    def sample(self, batch_size: int) -> List[Tuple[State, int, float, State, bool]]:
        pass

    @abstractmethod
    def __len__(self):
        pass

