from .atari_wrappers import LazyFrames, make_atari, wrap_deepmind
from .ext import Action, EnvExt, State
from .monitor import RewardMonitor
from .parallel import DummyParallelEnv, make_parallel_env, MultiProcEnv, ParallelEnv
from .parallel import FrameStackParallel, ParallelEnvWrapper
import numpy as np
from numpy import ndarray
import gym
from gym.spaces import Box
from typing import Tuple, Union


class ClassicalControl(EnvExt):
    def __init__(self, name: str = 'CartPole-v0', max_steps: int = 200) -> None:
        self.name = name
        super().__init__(gym.make(name))
        self._env._max_episode_steps = max_steps

    @property
    def action_dim(self) -> int:
        return self._env.action_space.n

    @property
    def state_dim(self) -> Tuple[int]:
        return self._env.observation_space.shape


class Atari(EnvExt):
    STYLES = ["deepmind", "baselines", "dopamine"]

    def __init__(self, name: str, style: str = "deepmind") -> None:
        assert style in self.STYLES, \
            'You have to choose a style from {}'.format(self.STYLES)
        if style is "dopamine":
            env = make_atari(name, timelimit=False, sticky_actions=True, noop_reset=False)
        else:
            env = make_atari(name)
        env = RewardMonitor(env)
        if style is "dopamine":
            env = wrap_deepmind(env, episodic_life=False, clip_rewards=False)
        elif style is "baselines":
            env = wrap_deepmind(env, fire_reset=True)
        else:
            env = wrap_deepmind(env)
        env = TransposeImage(env)
        super().__init__(env)

    @property
    def action_dim(self) -> int:
        return self._env.action_space.n

    @property
    def state_dim(self) -> Tuple[int]:
        return self._env.observation_space.shape

    def state_to_array(self, obs: State) -> ndarray:
        if type(obs) is LazyFrames:
            return obs.__array__()  # type: ignore
        else:
            return obs


class TransposeImage(gym.ObservationWrapper):
    """Transpose & scale image to use with Pytorch's CNN.
    Based on https://github.com/ikostrikov/pytorch-a2c-ppo-acktr, thanks:)
    """
    def __init__(self, env: gym.Env, scale: float = 255.0) -> None:
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.scale = scale
        self.observation_space: gym.Box = Box(
            self.observation_space.low[0, 0, 0],
            self.observation_space.high[0, 0, 0] / self.scale,
            [obs_shape[2], obs_shape[1], obs_shape[0]],
            dtype=self.observation_space.dtype
        )

    def observation(self, observation: Union[ndarray, LazyFrames]):
        t = type(observation)
        if t is LazyFrames:
            img = np.concatenate(observation._frames, axis=2).transpose(2, 0, 1)
        else:
            img = observation.transpose(2, 0, 1)  # type: ignore
        return img / self.scale
