from abc import ABC, abstractmethod
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from numpy import ndarray
import gym
from gym.spaces import Box
from torch import Tensor
from typing import Any, Generic, Tuple, TypeVar, Union

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
    def state_dims(self) -> Tuple[int]:
        pass

    def reset(self) -> State:
        return self._env.reset()

    def seed(self, seed: int) -> None:
        self._env.seed(seed)

    def step(self, action: Action) -> Tuple[State, float, bool, Any]:
        if type(action) == Tensor:
            action = action.item()
        return self._env.step(action)


class ClassicalControl(EnvExt):
    def __init__(self, name: str = 'CartPole-v0', max_steps: int = 200) -> None:
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
    def state_dims(self) -> Tuple[int]:
        return self.__env.observation_space.shape


class Atari(EnvExt):
    def __init__(
            self,
            name: str,
            clip_rewards: bool = True,
            episode_life: bool = True,
            frame_stack: bool = False,
    ) -> None:
        name += 'NoFrameskip-v4'
        env = make_atari(name)
        env = wrap_deepmind(
            env,
            episode_life=episode_life,
            clip_rewards=clip_rewards,
            frame_stack=frame_stack
        )
        env = wrap_pytorch(env)
        self.__env = env

    @property
    def _env(self) -> gym.Env:
        return self.__env

    @property
    def action_dim(self) -> int:
        return self.__env.action_space.n

    @property
    def state_dims(self) -> Tuple[int]:
        return self.__env.observation_space.shape


def wrap_pytorch(env: gym.Env) -> gym.Env:
    obs_shape = env.observation_space.shape
    if len(obs_shape) == 3 and obs_shape[2] in [1, 3]:
        env = TransposeImage(env)
    return env


# from https://github.com/ikostrikov/pytorch-a2c-ppo-acktr/blob/master/envs.py
class TransposeImage(gym.ObservationWrapper):
    def __init__(self, env: gym.Env = None) -> None:
        super(TransposeImage, self).__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0],
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype)

    def observation(self, observation: Union[Tensor, ndarray]):
        return observation.transpose(2, 0, 1)
