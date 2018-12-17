from abc import ABC, abstractmethod


class Cooler(ABC):
    @abstractmethod
    def __call__(self, eps: float) -> float:
        pass


class LinearCooler(Cooler):
    """decrease epsilon linearly, from initial to minimal, via `max_step` steps
    """
    def __init__(self, initial: float, minimal: float, max_step: int) -> None:
        self.delta = (initial - minimal) / float(max_step)
        self.minimal = minimal

    def __call__(self, eps: float) -> float:
        return max(eps - self.delta, self.minimal)


class DummyCooler(Cooler):
    """Do nothing
    """
    def __init__(self, *args) -> None:
        pass

    def __call__(self, eps: float) -> float:
        return eps


