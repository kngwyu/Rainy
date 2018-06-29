from abc import ABC, abstractmethod
from numpy import ndarray


class Policy(ABC):
    # TODO: continuous
    @abstractmethod
    def select_action(self, state: ndarray) -> int:
        pass

    @abstractmethod
    def update(self) -> None:
        pass


