from typing import Tuple, Union

import gym
import numpy as np
from gym.spaces import Box
from gym.wrappers import TimeLimit

from ..prelude import Array
from ..utils import RunningMeanStd
from .atari_wrappers import LazyFrames


class TransposeObs(gym.ObservationWrapper):
    """Transpose & Scale image
    """

    def __init__(
        self,
        env: gym.Env,
        transpose: Tuple[int, int, int] = (2, 0, 1),
        scale: float = 255.0,
    ) -> None:
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space: gym.Box = Box(
            low=self.observation_space.low[0, 0, 0],
            high=self.observation_space.high[0, 0, 0] / scale,
            shape=[obs_shape[i] for i in transpose],
            dtype=self.observation_space.dtype,
        )
        self.scale = scale
        self.transpose = transpose

    def observation(self, observation: Union[Array, LazyFrames]):
        if isinstance(observation, LazyFrames):
            img = np.concatenate(observation._frames, axis=2).transpose(2, 0, 1)
        else:
            img = observation.transpose(*self.transpose)
        return img / self.scale


class ScaleObs(gym.ObservationWrapper):
    """Transpose & Scale image
    """

    def __init__(self, env: gym.Env, scale: float = 255.0) -> None:
        super().__init__(env)
        self.scale = scale

    def observation(self, obs: Array):
        return obs / self.scale


class AddTimeStep(gym.ObservationWrapper):
    def __init__(self, env: TimeLimit) -> None:
        super().__init__(env)
        obs = self.observation_space
        self.observation_space: gym.Box = Box(
            low=obs.low[0], high=obs.high[0], shape=[obs.shape[0] + 1], dtype=obs.dtype
        )

    def observation(self, obs: Array[float]) -> Array[float]:
        return np.append(obs, self.env._elapsed_steps)


class NormalizeObs(gym.ObservationWrapper):
    def __init__(self, env: gym.Env, obs_clip: float = 10.0) -> None:
        super().__init__(env)
        self.obs_clip = obs_clip
        self._rms = RunningMeanStd(shape=self.observation_space.shape)
        self._training_mode = False

    def observation(self, obs: Array[float]) -> Array[float]:
        if self._training_mode:
            self._rms.update(obs)
        return np.clip(
            (obs - self._rms.mean) / self._rms.std(), -self.obs_clip, self.obs_clip
        )
