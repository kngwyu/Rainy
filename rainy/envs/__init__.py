from numpy import ndarray
import gym
from typing import Callable, Optional
from .atari_wrappers import LazyFrames, make_atari, wrap_deepmind
from .deepsea import DeepSea as DeepSeaGymEnv
from .ext import EnvExt, EnvSpec
from .monitor import RewardMonitor
from .obs_wrappers import AddTimeStep, TransposeObs
from .parallel import DummyParallelEnv, EnvGen, MultiProcEnv, ParallelEnv
from .parallel_wrappers import (
    FrameStackParallel,
    NormalizeObs,
    NormalizeReward,
    ParallelEnvWrapper,
)
from ..prelude import Self, State


class AtariConfig:
    STYLES = ["deepmind", "baselines", "dopamine", "rnd"]

    def __init__(self) -> None:
        self.timelimit = True
        self.override_timelimit: Optional[int] = None
        self.v4 = False
        self.sticky_actions = False
        self.noop_reset = True
        self.episodic_life = False
        self.fire_reset = False
        self.clip_reward = True
        self.flicker_frame = False
        self.frame_stack = True

    @staticmethod
    def from_style(style: str) -> Self:
        if style not in AtariConfig.STYLES:
            raise ValueError(
                "You have to choose a style from {}".format(AtariConfig.STYLES)
            )
        cfg = AtariConfig()
        if style == "deepmind":
            pass
        elif style == "baselines":
            cfg.fire_reset = True
        elif style == "dopamine":
            cfg.timelimit = False
            cfg.v4 = True
            cfg.noop_reset = False
            cfg.episodic_life = True
        elif style == "rnd":
            cfg.override_timelimit = 4500 * 4
            cfg.noop_reset = False
            cfg.sticky_actions = True
            cfg.v4 = True
        return cfg


class Atari(EnvExt):
    def __init__(self, name: str, cfg: AtariConfig = AtariConfig(), **kwargs) -> None:
        cfg.__dict__.update(kwargs)
        env = make_atari(
            name,
            timelimit=cfg.timelimit,
            override_timelimit=cfg.override_timelimit,
            v4=cfg.v4,
            sticky_actions=cfg.sticky_actions,
            noop_reset=cfg.noop_reset,
        )
        env = RewardMonitor(env)
        env = wrap_deepmind(
            env,
            episodic_life=cfg.episodic_life,
            fire_reset=cfg.fire_reset,
            clip_reward=cfg.clip_reward,
            flicker_frame=cfg.flicker_frame,
            frame_stack=cfg.frame_stack,
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


class ClassicControl(EnvExt):
    def __init__(
        self, name: str = "CartPole-v0", max_steps: Optional[int] = None
    ) -> None:
        self.name = name
        super().__init__(gym.make(name))
        if max_steps is not None:
            self._env._max_episode_steps = max_steps


class DeepSea(EnvExt):
    def __init__(self, size: int, noise: float = 0.0) -> None:
        env = DeepSeaGymEnv(size, noise)
        super().__init__(env)


class PyBullet(EnvExt):
    def __init__(
        self, name: str = "Hopper", add_timestep: bool = False, nosuffix: bool = False
    ) -> None:
        self.name = name
        try:
            import pybullet_envs  # noqa
        except ImportError:
            raise ImportError("pybullet is not installed")
        if not nosuffix:
            name += "BulletEnv-v0"
        env = gym.make(name)
        if add_timestep:
            env = AddTimeStep(env)
        super().__init__(RewardMonitor(env))
        self.viewer = None
        self.spec.use_reward_monitor = True

    def render(self, mode: str = "human") -> Optional[ndarray]:
        if mode == "human":
            arr = self._env.render("rgb_array")
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
    gamma: float = 0.99,
) -> Callable[[EnvGen, int], ParallelEnv]:
    def __wrap(env_gen: EnvGen, num_workers: int) -> ParallelEnv:
        penv: ParallelEnv = MultiProcEnv(env_gen, num_workers)
        if normalize_obs:
            penv = NormalizeObs(penv, obs_clip=obs_clip)
        if normalize_reward:
            penv = NormalizeReward(penv, reward_clip=reward_clip, gamma=gamma)
        return penv

    return __wrap


# Same as bsuite
gym.envs.register(
    id="CartPoleSwingUp-v0",
    entry_point="rainy.envs.swingup:CartPoleSwingUp",
    max_episode_steps=1000,
    kwargs=dict(start_position="bottom", allow_noop=True),
    reward_threshold=800,
)

# More difficult
gym.envs.register(
    id="CartPoleSwingUp-v1",
    entry_point="rainy.envs.swingup:CartPoleSwingUp",
    max_episode_steps=1000,
    kwargs=dict(start_position="bottom", allow_noop=True, height_threshold=0.9),
    reward_threshold=800,
)


# No movecost
gym.envs.register(
    id="CartPoleSwingUp-v2",
    entry_point="rainy.envs.swingup:CartPoleSwingUp",
    max_episode_steps=1000,
    kwargs=dict(start_position="bottom", allow_noop=False),
    reward_threshold=900,
)

# Arbitary start
gym.envs.register(
    id="CartPoleSwingUp-v3",
    entry_point="rainy.envs.swingup:CartPoleSwingUp",
    max_episode_steps=1000,
    kwargs=dict(start_position="arbitary", allow_noop=False),
    reward_threshold=900,
)
