from abc import ABC, abstractmethod
from numpy import ndarray
from typing import Callable, Union


class Explorer(ABC):
    @abstractmethod
    def select_action(self, state: ndarray) -> int:
        pass



