from abc import ABC, abstractmethod
from numpy import ndarray

class Explorer(ABC):
    @abstractmethod
    def select_action(self, state: ndarray) -> int:
        pass



