from abc import ABC, abstractmethod
import numpy as np
from torch import Tensor
from torch.optim import Optimizer
from typing import Callable
from ..net.value import QFunction
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
    def select_action(self, state: Array, qfunc: QFunction) -> int:
        return self._select_from_fn(lambda: qfunc.action_values(state).detach(), qfunc.action_dim)

    def select_from_value(self, value: Tensor) -> int:
        return self._select_from_fn(lambda: value)

    @abstractmethod
    def _select_from_fn(self, value_fn: Callable[[], Tensor], action_dim: int) -> int:
        pass


class Greedy(Explorer):
    """deterministic greedy policy
    """
    def _select_from_fn(self, value_fn: Callable[[], Tensor], _action_dim: int) -> int:
        return value_fn().argmax().item()


class EpsGreedy(Explorer):
    """ε-greedy policy
    """
    def __init__(self, epsilon: float, cooler: Cooler) -> None:
        self.epsilon = epsilon
        self.cooler = cooler

    def _select_from_fn(self, value_fn: Callable[[], Tensor], action_dim: int) -> int:
        old_eps = self.epsilon
        self.epsilon = self.cooler()
        if np.random.rand() < old_eps:
            return np.random.randint(0, action_dim)
        return value_fn().argmax().item()
