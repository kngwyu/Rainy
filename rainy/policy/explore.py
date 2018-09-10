"""Exploration strategies for value based algorithms
"""
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
from .base import Policy
from .cooler import Cooler, LinearCooler
from ..net.value_net import ValueNet


class Explorer(Policy, ABC):
    @abstractmethod
    def update(self) -> None:
        pass


class Greedy(Policy):
    """ε-greedy policy
    """
    def __init__(self, epsilon: float, cooler: Cooler, value_net: ValueNet) -> None:
        self.epsilon = epsilon
        self.cooler = cooler
        self.value_net = value_net

    def select_action(self, state: ndarray) -> int:
        if np.random.rand() < self.epsilon:
            action_dim = self.value_net.action_dim
            return np.random.randint(0, action_dim)
        action_values = self.valuenet.action_values(state)
        return action_values.argmax()

    def update(self) -> None:
        self.epsilon = self.cooler(self.epsilon)


def linear_greedy(epsilon: float, value_net: ValueNet) -> Greedy:
    Greedy(epsilon, LinearCooler, value_net)
