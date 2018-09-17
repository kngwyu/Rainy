import gym
from abc import ABC, abstractmethod
from typing import Any, Generic, Tuple, TypeVar

class EnvExt(gym.env, ABC):
    @abstractmethod
    @property
    def action_dim(self) -> int:
        pass

    @abstractmethod
    @property
    def state_dim(self) -> int:
        pass


class ClassicalControl(EnvExt):
    def __init__(self, name: str = 'CartPole-v0', max_steps: int = 200):
        self.__dict__ = gym.make(self.name).__dict__
