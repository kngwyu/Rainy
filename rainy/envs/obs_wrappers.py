import gym
from gym.wrappers import TimeLimit
from gym.spaces import Box
import numpy as np
from typing import Tuple, Union
from .atari_wrappers import LazyFrames
from ..prelude import Array


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


class AddTimeStep(gym.ObservationWrapper):
    def __init__(self, env: TimeLimit) -> None:
        super().__init__(env)
        obs = self.observation_space
        self.observation_space: gym.Box = Box(
            low=obs.low[0], high=obs.high[0], shape=[obs.shape[0] + 1], dtype=obs.dtype
        )

    def observation(self, obs: Array[float]) -> Array[float]:
        return np.append(obs, self.env._elapsed_steps)
