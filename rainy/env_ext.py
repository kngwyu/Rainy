import gym
from abc import ABC, abstractmethod
from typing import Any, Generic, Tuple, TypeVar

Action = TypeVar('Action')
State = TypeVar('State')


class EnvExt(gym.Env, ABC, Generic[Action, State]):
    @property
    @abstractmethod
    def __inner(self) -> gym.Env:
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def state_dim(self) -> int:
        pass

    def reset(self) -> State:
        return self.__inner.reset()

    def step(self, action: Action) -> Tuple[State, float, bool, Any]:
        return self.__inner.step(action)

    def seed(self, seed: int) -> None:
        self.seed(seed)


class ClassicalControl(EnvExt):
    def __init__(self, name: str = 'CartPole-v0', max_steps: int = 200):
        self.name = name
        self.__dict__ = gym.make(name).__dict__
        self._max_episode_steps = max_steps

    @property
    def action_dim(self) -> int:
        return self.action_space.n

    @property
    def state_dim(self) -> int:
        return self.observation_space.shape[0]
