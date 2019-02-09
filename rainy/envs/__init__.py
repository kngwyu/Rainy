from .atari_wrappers import LazyFrames, make_atari, wrap_deepmind
from .ext import Action, EnvExt, EnvSpec, State
from .monitor import RewardMonitor
from .parallel import DummyParallelEnv, MultiProcEnv, ParallelEnv
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


class Atari(EnvExt):
    STYLES = ['deepmind', 'baselines', 'dopamine']

    def __init__(self, name: str, style: str = 'deepmind', frame_stack: bool = True) -> None:
        assert style in self.STYLES, \
            'You have to choose a style from {}'.format(self.STYLES)
        if style is 'dopamine':
            env = make_atari(name, timelimit=False, sticky_actions=True, noop_reset=False)
        else:
            env = make_atari(name)
        env = RewardMonitor(env)
        env = wrap_deepmind(
            env,
            episodic_life=style is 'dopamine',
            fire_reset=style is 'baselines',
            frame_stack=frame_stack
        )
        env = TransposeObs(env)
        super().__init__(env)
        self.spec = EnvSpec(*self.spec[:2], True)

    def state_to_array(self, obs: State) -> ndarray:
        if type(obs) is LazyFrames:
            return obs.__array__()  # type: ignore
        else:
            return obs


class TransposeObs(gym.ObservationWrapper):
    """Transpose & Scale image
    """
    def __init__(
            self,
            env: gym.Env,
            transpose: Tuple[int, int, int] = (2, 0, 1),
            scale: float = 255.0
    ) -> None:
        super().__init__(env)
        obs_shape = self.observation_space.shape
        self.observation_space: gym.Box = Box(
            low=self.observation_space.low[0, 0, 0],
            high=self.observation_space.high[0, 0, 0] / scale,
            shape=[obs_shape[i] for i in transpose],
            dtype=self.observation_space.dtype
        )
        self.scale = scale
        self.transpose = transpose

    def observation(self, observation: Union[ndarray, LazyFrames]):
        t = type(observation)
        if t is LazyFrames:
            img = np.concatenate(observation._frames, axis=2).transpose(2, 0, 1)
        else:
            img = observation.transpose(*self.transpose)  # type: ignore
        return img / self.scale
