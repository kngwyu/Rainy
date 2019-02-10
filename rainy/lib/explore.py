from abc import ABC, abstractmethod
import numpy as np
from torch.optim import Optimizer
from ..net.value import ValuePredictor
from ..utils.typehack import Array


class Cooler(ABC):
    @abstractmethod
    def __call__(self, eps: float) -> float:
        pass

    def lr_decay(self, optim: Optimizer) -> None:
        for param_group in Optimizer.param_groups:
            if 'lr' in param_group:
                param_group['lr'] = self.__call__(param_group['lr'])


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
        self.epsilon = self.cooler(self.epsilon)
        if np.random.rand() < old_eps:
            action_dim = value_pred.action_dim
            return np.random.randint(0, action_dim)
        return value_pred.action_values(state).detach().argmax().item()
