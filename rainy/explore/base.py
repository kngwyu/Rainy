from abc import ABC, abstractmethod
from numpy import ndarray

class Exploler(ABC):
    @abstractmethod
    def select_action(self, state: ndarray) -> int:
        pass



