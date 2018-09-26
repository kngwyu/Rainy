"""Exploration strategies for value based algorithms
"""
import numpy as np
from numpy import ndarray
from .base import Explorer
from .cooler import Cooler
from ..net.value_net import ValueNet


class Greedy(Explorer):
    """deterministic greedy policy
    """
    def __init__(self, value_net: ValueNet) -> None:
        self.value_net = value_net

    def select_action(self, state: ndarray) -> int:
        action_values = self.value_net.action_values(state).detach()
        return action_values.argmax()


class EpsGreedy(Explorer):
    """ε-greedy policy
    """
    def __init__(self, epsilon: float, cooler: Cooler, value_net: ValueNet) -> None:
        self.epsilon = epsilon
        self.cooler = cooler
        self.value_net = value_net

    def select_action(self, state: ndarray) -> int:
        old_eps = self.epsilon
        self.epsilon = self.cooler(self.epsilon)
        if np.random.rand() < old_eps:
            action_dim = self.value_net.action_dim
            return np.random.randint(0, action_dim)
        action_values = self.value_net.action_values(state).detach()
        return action_values.argmax()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state['value_net']
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
