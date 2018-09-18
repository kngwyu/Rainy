"""Exploration strategies for value based algorithms
"""
from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
from .base import Explorer
from .cooler import Cooler, LinearCooler
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
        if np.random.rand() < self.epsilon:
            action_dim = self.value_net.action_dim
            return np.random.randint(0, action_dim)
        self.epsilon = self.cooler(self.epsilon)
        action_values = self.value_net.action_values(state).detach()
        return action_values.argmax()

