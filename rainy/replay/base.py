from abc import ABC, abstractmethod

class BaseReplay(ABC):
    @abstractmethod
    def append(self):
        pass
