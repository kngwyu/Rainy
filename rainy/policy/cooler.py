from abc import ABC, abstractmethod

class Cooler(ABC):
    @abstractmethod
    def __call__(self, eps: float) -> float:
        pass

class LinearCooler(Cooler):
    """decrease epsilon linearly, from initial to minimal, via `max_step` steps
    """
    def __init__(self, initial: float, minimal: float, max_step: int) -> None:
        self.initial = initial
        self.minimal = minimal
        self.max_step = max_step
        self.current_step = 0

    def __call__(self, eps: float) -> float:
        part = float(self.current_step) / float(self.terminate_step)
        self.current_step += 1
        res = eps - part * (self.initial - self.minimal)
        return max(res, self.minimal)



