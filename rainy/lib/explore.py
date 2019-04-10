from abc import ABC, abstractmethod
import numpy as np
from torch.optim import Optimizer
from ..net.value import ValuePredictor
from ..prelude import Array


class Cooler(ABC):
    @abstractmethod
    def __call__(self) -> float:
        pass

    def lr_decay(self, optim: Optimizer) -> None:
        for param_group in optim.param_groups:
            if 'lr' in param_group:
                param_group['lr'] = self.__call__()


class LinearCooler(Cooler):
    """decrease epsilon linearly, from initial to minimal, via `max_step` steps
    """
    def __init__(self, initial: float, minimal: float, max_step: int) -> None:
        if initial < minimal:
            raise ValueError(
                f'[LinearCooler.__init__] the minimal value({minimal})'
                ' is bigger than the initial value {initial}'
            )
        self.base = initial - minimal
        self.minimal = minimal
        self.max_step = max_step
        self.left = max_step

    def __call__(self) -> float:
        self.left = max(0, self.left - 1)
        return (float(self.left) / self.max_step) * self.base + self.minimal


class DummyCooler(Cooler):
    """Do nothing
    """
    def __init__(self, initial: float, *args) -> None:
        self.initial = initial

    def __call__(self) -> float:
        return self.initial


class Explorer(ABC):
    @abstractmethod
    def select_action(self, state: Array, value_pred: ValuePredictor) -> int:
        pass


class Greedy(Explorer):
    """deterministic greedy policy
    """
    def select_action(self, state: Array, value_pred: ValuePredictor) -> int:
        return value_pred.action_values(state).detach().argmax().item()


class EpsGreedy(Explorer):
    """ε-greedy policy
    """
    def __init__(self, epsilon: float, cooler: Cooler) -> None:
        self.epsilon = epsilon
        self.cooler = cooler

    def select_action(self, state: Array, value_pred: ValuePredictor) -> int:
        old_eps = self.epsilon
        self.epsilon = self.cooler()
        if np.random.rand() < old_eps:
            action_dim = value_pred.action_dim
            return np.random.randint(0, action_dim)
        return value_pred.action_values(state).detach().argmax().item()
