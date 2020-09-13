from typing import Callable, Optional, Union

import gym
import numpy as np

from ..prelude import Array, Self, State
from .atari_wrappers import LazyFrames, make_atari, wrap_deepmind
from .deepsea import DeepSea as DeepSeaGymEnv
from .ext import EnvExt, EnvSpec, EnvTransition  # noqa
from .monitor import RewardMonitor
from .obs_wrappers import AddTimeStep, NormalizeObs, ScaleObs, TransposeObs  # noqa
from .parallel import (  # noqa
    DummyParallelEnv,
    EnvGen,
    MultiProcEnv,
    ParallelEnv,
    PEnvTransition,
)
from .parallel_wrappers import (  # noqa
    FrameStackParallel,
    NormalizeObsParallel,
    NormalizeRewardParallel,
    ParallelEnvWrapper,
)


class AtariConfig:
    """
    Configuration of Atari wrappers.
    By default, 'deepmind' config is used.
    """

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
    def __init__(
        self, name: str, cfg: Union[AtariConfig, str] = "deepmind", **kwargs
    ) -> None:
        if isinstance(cfg, str):
            cfg = AtariConfig.from_style(cfg)
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

    @staticmethod
    def extract(obs: State) -> Array:
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


class RLPyGridWorld(EnvExt):
    """
    Class for RLPy3 grid world.
    Current gym API cannot allow `reset` to pass raw observation, so here
    we use this wrapper.
    """

    # To make CLI options shorter, we have a set of aliases of environment.
    ALIASES = {
        "2RoomsEasy": "RLPyFixedRewardGridWorld9x11-2RoomsEasy",
        "2RoomsSparse": "RLPyFixedRewardGridWorld9x11-2RoomsSparse",
        "2RoomsExp": "RLPyFixedRewardGridWorld9x11-2Rooms",
        "4Rooms": "RLPyGridWorld11x11-4Rooms-RandomGoal",
        "4RoomsEasy": "RLPyFixedRewardGridWorld11x11-4RoomsEasy",
        "4RoomsExp": "RLPyFixedRewardGridWorld11x11-4Rooms",
        "4RoomsBer": "RLPyBernoulliGridWorld11x11-4Rooms",
        "9RoomsExp": "RLPyFixedRewardGridWorld17x17-9Rooms",
    }

    def __init__(
        self, name: str = "4Rooms", obs_type: str = "image", max_steps: int = 100,
    ) -> None:
        try:
            from rlpy.gym import gridworld_obs
        except ImportError:
            raise ImportError("RLPy3 is not installed")

        envname = self.ALIASES.get(name, name)
        env = gym.make(envname + "-v1")
        if max_steps is not None:
            env._max_episode_steps = max_steps
        self.domain = env.unwrapped.domain  # Get RLPy domain
        obs_fn, obs_space = gridworld_obs(self.domain, mode=obs_type)
        super().__init__(env, obs_space.shape)
        self.obs_fn = obs_fn
        self.obs_type = obs_type

    def extract(self, state: Array[int]) -> Array[float]:
        return self.obs_fn(self.domain, state[:2].astype(int))


class PyBullet(EnvExt):
    """PyBullet  environment.
    """

    def __init__(
        self, name: str = "Hopper", add_timestep: bool = False, nosuffix: bool = False
    ) -> None:
        self.name = name
        try:
            import pybullet_envs  # noqa
        except ImportError:
            raise ImportError("PyBullet is not installed")
        if not nosuffix:
            name += "BulletEnv-v0"
        env = gym.make(name)
        if add_timestep:
            env = AddTimeStep(env)
        super().__init__(RewardMonitor(env))
        self._viewer = None

    def render(self, mode: str = "human") -> Optional[Array]:
        arr = self._env.render("rgb_array").astype(np.uint8)
        if mode == "human":
            if self._viewer is None:
                from gym.envs.classic_control import rendering

                self._viewer = rendering.SimpleImageViewer()
            self._viewer.imshow(arr)
            return None
        else:
            return arr

    def close(self) -> None:
        if self._viewer is not None:
            self._viewer.close()


class Mujoco(EnvExt):
    def __init__(self, name: str = "Hopper-v2", add_timestep: bool = False) -> None:
        self.name = name
        try:
            import mujoco_py  # noqa
        except ImportError:
            raise ImportError("mujoco_py is not installed")
        if "-v" not in name:
            name += "-v2"
        env = gym.make(name)
        if add_timestep:
            env = AddTimeStep(env)
        super().__init__(RewardMonitor(env))


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
            penv = NormalizeObsParallel(penv, obs_clip=obs_clip)
        if normalize_reward:
            penv = NormalizeRewardParallel(penv, reward_clip=reward_clip, gamma=gamma)
        return penv

    return __wrap


SWINGUP_PARAMS = [
    # Same as bsuite
    dict(start_position="bottom", allow_noop=True),
    # Difficult
    dict(start_position="bottom", allow_noop=True, height_threshold=0.9),
    # No movecost
    dict(start_position="bottom", allow_noop=False),
    # Easy
    dict(
        start_position="bottom",
        allow_noop=False,
        height_threshold=0.0,
        theta_dot_threshold=1.5,
        x_reward_threshold=1.5,
    ),
    # Arbitary start
    dict(start_position="arbitary", allow_noop=False),
]


for i, param in enumerate(SWINGUP_PARAMS):
    gym.envs.register(
        id=f"CartPoleSwingUp-v{i}",
        entry_point="rainy.envs.cartpole_ext:CartPoleSwingUp",
        max_episode_steps=1000,
        kwargs=param,
        reward_threshold=800,
    )
    gym.envs.register(
        id=f"CartPoleSwingUpContinuous-v{i}",
        entry_point="rainy.envs.cartpole_ext:CartPoleSwingUpContinuous",
        max_episode_steps=1000,
        kwargs=param,
        reward_threshold=800,
    )
