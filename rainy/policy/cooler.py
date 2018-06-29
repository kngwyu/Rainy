from abc import ABC, abstractmethod

class Cooler(ABC):
    @abstractmethod
    def __call__(self, eps: float) -> float:
        pass

class LinearCooler(Cooler):
    def __init__(self, initial: float, minimal: float, terminate: int) -> None:
        self.initial = initial
        self.minimal = minimal
        self.terminate = terminate
        self.current = 0

    def __call__(self, eps: float) -> float:
        part = float(self.current) / float(self.terminate)
        self.current += 1
        res = eps - part * (self.initial - self.minimal)
        return max(res, self.minimal)



