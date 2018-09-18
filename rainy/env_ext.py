import gym
from abc import ABC, abstractmethod
from torch import Tensor
from typing import Any, Generic, Tuple, TypeVar

Action = TypeVar('Action')
State = TypeVar('State')


class EnvExt(gym.Env, ABC, Generic[Action, State]):
    @property
    @abstractmethod
    def _env(self) -> gym.Env:
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
        return self._env.reset()

    def step(self, action: Action) -> Tuple[State, float, bool, Any]:
        if type(action) == Tensor:
            action = action.item()
        return self._env.step(action)

    def seed(self, seed: int) -> None:
        self._env.seed(seed)


class ClassicalControl(EnvExt):
    def __init__(self, name: str = 'CartPole-v0', max_steps: int = 200):
        self.name = name
        self.__env = gym.make(name)
        self.__env._max_episode_steps = max_steps

    @property
    def _env(self) -> gym.Env:
        return self.__env

    @property
    def action_dim(self) -> int:
        return self.__env.action_space.n

    @property
    def state_dim(self) -> int:
        return self.__env.observation_space.shape[0]
