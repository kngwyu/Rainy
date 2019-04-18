from numpy import ndarray
import gym
from typing import Callable, Optional
from .atari_wrappers import LazyFrames, make_atari, wrap_deepmind
from .ext import EnvExt, EnvSpec
from .monitor import RewardMonitor
from .obs_wrappers import AddTimeStep, TransposeObs
from .parallel import DummyParallelEnv, EnvGen, MultiProcEnv, ParallelEnv
from .parallel_wrappers import FrameStackParallel, NormalizeObs, \
    NormalizeReward, ParallelEnvWrapper
from ..prelude import State


class Atari(EnvExt):
    STYLES = ['deepmind', 'baselines', 'dopamine']

    def __init__(
            self,
            name: str,
            style: str = 'deepmind',
            flicker_frame: bool = False,
            frame_stack: bool = True,
    ) -> None:
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
        self.spec.use_reward_monitor = True

    @staticmethod
    def extract(obs: State) -> ndarray:
        if type(obs) is LazyFrames:
            return obs.__array__()  # type: ignore
        else:
            return obs


def atari_parallel(frame_stack: bool = True) -> Callable[[EnvGen, int], ParallelEnv]:
    def __wrap(env_gen: EnvGen, num_workers: int) -> ParallelEnv:
        penv: ParallelEnv = MultiProcEnv(env_gen, num_workers)
        if frame_stack:
            penv = FrameStackParallel(penv)
        return penv
    return __wrap


class ClassicalControl(EnvExt):
    def __init__(self, name: str = 'CartPole-v0', max_steps: int = 200) -> None:
        self.name = name
        super().__init__(gym.make(name))
        self._env._max_episode_steps = max_steps


class PyBullet(EnvExt):
    def __init__(
            self,
            name: str = 'Hopper',
            add_timestep: bool = False,
            nosuffix: bool = False
    ) -> None:
        self.name = name
        try:
            import pybullet_envs  # noqa
        except ImportError:
            raise ImportError('pybullet is not installed')
        if not nosuffix:
            name += 'BulletEnv-v0'
        env = gym.make(name)
        if add_timestep:
            env = AddTimeStep(env)
        super().__init__(RewardMonitor(env))
        self.viewer = None
        self.spec.use_reward_monitor = True

    def render(self, mode: str = 'human') -> Optional[ndarray]:
        if mode == 'human':
            arr = self._env.render('rgb_array')
            if self.viewer is None:
                from gym.envs.classic_control.rendering import SimpleImageViewer
                self.viewer = SimpleImageViewer()
            self.viewer.imshow(arr)  # type: ignore
            return None
        else:
            return self._env.render(mode)


def pybullet_parallel(
        normalize_obs: bool = True,
        normalize_reward: bool = True,
        obs_clip: float = 10.0,
        reward_clip: float = 10.0,
        gamma: float = 0.99
) -> Callable[[EnvGen, int], ParallelEnv]:
    def __wrap(env_gen: EnvGen, num_workers: int) -> ParallelEnv:
        penv: ParallelEnv = MultiProcEnv(env_gen, num_workers)
        if normalize_obs:
            penv = NormalizeObs(penv, obs_clip=obs_clip)
        if normalize_reward:
            penv = NormalizeReward(penv, reward_clip=reward_clip, gamma=gamma)
        return penv
    return __wrap

