from abc import ABC, abstractmethod
from .atari_wrappers import LazyFrames, make_atari, wrap_deepmind
import numpy as np
from numpy import ndarray
import gym
from gym.spaces import Box
from torch import Tensor
from typing import Any, Callable, Generic, Tuple, TypeVar, Union


Action = TypeVar('Action', int, Tensor)
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
        if type(action) == Tensor:
            return self._env.step(action.item())
        else:
            return self._env.step(action)

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
    def state_dims(self) -> Tuple[int]:
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


EnvGen = Callable[[], EnvExt]


class ClassicalControl(EnvExt):
    def __init__(self, name: str = 'CartPole-v0', max_steps: int = 200) -> None:
        self.name = name
        super().__init__(gym.make(name))
        self._env._max_episode_steps = max_steps

    @property
    def action_dim(self) -> int:
        return self._env.action_space.n

    @property
    def state_dims(self) -> Tuple[int]:
        return self._env.observation_space.shape


class Atari(EnvExt):
    def __init__(
            self,
            name: str,
            clip_rewards: bool = True,
            episode_life: bool = True,
            frame_stack: bool = True,
    ) -> None:
        name += 'NoFrameskip-v4'
        env = make_atari(name)
        env = wrap_deepmind(
            env,
            episode_life=episode_life,
            clip_rewards=clip_rewards,
            frame_stack=frame_stack
        )
        env = TransposeImage(env)
        super().__init__(env)

    @property
    def action_dim(self) -> int:
        return self._env.action_space.n

    @property
    def state_dims(self) -> Tuple[int]:
        return self._env.observation_space.shape


# based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env: gym.Env = None) -> None:
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.scaler = np.vectorize(lambda x: x / 255.0)
        self.observation_space: gym.Box = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[0], obs_shape[1]],
            dtype=self.observation_space.dtype)

    def observation(self, observation: Union[ndarray, LazyFrames]):
        t = type(observation)
        if t is LazyFrames:
            img = np.concatenate(observation._frames, axis=2).transpose(2, 0, 1)
        else:
            img = observation.transpose(2, 0, 1)
        return self.scaler(img)
