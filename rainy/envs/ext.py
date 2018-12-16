from abc import ABC, abstractmethod
import gym
from numpy import ndarray
from typing import Any, Generic, Tuple, TypeVar


Action = TypeVar('Action', bound=int)
State = TypeVar('State')


class EnvExt(gym.Env, ABC, Generic[Action, State]):
    def __init__(self, env: gym.Env) -> None:
        self._env = env

    def close(self):
        """
        Inherited from gym.Env.
        """
        self._env.close

    def reset(self) -> State:
        """
        Inherited from gym.Env.
        """
        return self._env.reset()

    def render(self, mode: str = 'human') -> None:
        """
        Inherited from gym.Env.
        """
        self._env.render(mode=mode)

    def seed(self, seed: int) -> None:
        """
        Inherited from gym.Env.
        """
        self._env.seed(seed)

    def step(self, action: Action) -> Tuple[State, float, bool, Any]:
        """
        Inherited from gym.Env.
        """
        return self._env.step(action)

    def step_and_reset(self, action: Action) -> Tuple[State, float, bool, Any]:
        state, reward, done, info = self.step(action)
        if done:
            state = self.reset()
        return state, reward, done, info

    @property
    def unwrapped(self) -> gym.Env:
        """
        Inherited from gym.Env.
        """
        return self._env.unwrapped

    @property
    @abstractmethod
    def action_dim(self) -> int:
        """
        Extended method.
        Returns a ndim of action space.
        """
        pass

    @property
    @abstractmethod
    def state_dim(self) -> Tuple[int]:
        """
        Extended method.
        Returns a shape of observation space.
        """
        pass

    def state_to_array(self, state: State) -> ndarray:
        """
        Extended method.
        Convert state to ndarray.
        It's useful for the cases where numpy.ndarray representation is too large to
        throw it to replay buffer directly.
        """
        return state

    def save_history(self, file_name: str) -> None:
        """
        Extended method.
        Save agent's action history to file.
        """
        pass
