from abc import ABC, abstractmethod


class Cooler(ABC):
    @abstractmethod
    def __call__(self, eps: float) -> float:
        pass


class LinearCooler(Cooler):
    """decrease epsilon linearly, from initial to minimal, via `max_step` steps
    """
    def __init__(self, initial: float, minimal: float, max_step: int) -> None:
        self.delta = (initial - minimal) / max_step
        self.minimal = minimal

    def __call__(self, eps: float) -> float:
        return max(eps - self.delta, self.minimal)



