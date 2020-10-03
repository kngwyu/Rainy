"""Environment specifications:
"""
from typing import Callable, Generic, NamedTuple, Optional, Sequence, Type, Union

import gym
import numpy as np
from gym import spaces

from ..prelude import Action, Array, GenericNamedMeta, Self, State
from .monitor import RewardMonitor


class EnvTransition(NamedTuple, Generic[State], metaclass=GenericNamedMeta):
    state: State
    reward: float
    terminal: bool
    info: dict

    def map_r(self, f: Callable[[float], float]) -> Self:
        s, r, t, i = self
        return EnvTransition(s, f(r), t, i)


class EnvSpec:
    """EnvSpec holds obs/action dims and monitors
    """

    def __init__(
        self,
        state_dim: Sequence[int],
        action_space: gym.Space,
        use_reward_monitor: bool = False,
    ) -> None:
        self.state_dim = state_dim
        self.action_space = action_space
        self.use_reward_monitor = use_reward_monitor
        if isinstance(action_space, spaces.Discrete):
            self.action_dim = action_space.n
            self._act_range = 0, action_space.n
        elif isinstance(action_space, spaces.Box):
            if len(action_space.shape) != 1:
                raise RuntimeError("Box space with shape >= 2 is not supportd")
            self.action_dim = action_space.shape[0]
            self._act_range = action_space.low, action_space.high
        else:
            raise RuntimeError("{} is not supported".format(type(action_space)))

    def clip_action(self, act: Array) -> Array:
        return np.clip(act, *self._act_range)

    def random_action(self) -> Action:
        return self.action_space.sample()

    def random_actions(self, n: int) -> Array[Action]:
        return np.array([self.action_space.sample() for _ in range(n)])

    def is_discrete(self) -> bool:
        return isinstance(self.action_space, spaces.Discrete)

    def __repr__(self) -> str:
        return "EnvSpec(state_dim: {} action_space: {})".format(
            self.state_dim, self.action_space
        )


class EnvExt(gym.Env, Generic[Action, State]):
    def __init__(self, env: gym.Env, obs_shape: Optional[spaces.Space] = None) -> None:
        self._env = env
        if obs_shape is None:
            obs_shape = env.observation_space.shape
            if obs_shape is None:
                raise NotImplementedError(
                    f"Failed detect state dimension from {env.obs_shape}!"
                )
        use_reward_monitor = _use_reward_monitor(env)
        self._spec = EnvSpec(obs_shape, self._env.action_space, use_reward_monitor)
        self._eval = False

    def close(self):
        """ Inherited from gym.Env.
        """
        self._env.close()

    def reset(self) -> State:
        """ Inherited from gym.Env.
        """
        return self._env.reset()

    def render(self, mode: str = "human") -> Optional[Array]:
        """ Inherited from gym.Env.
        """
        return self._env.render(mode=mode)

    def seed(self, seed: int) -> None:
        """ Inherited from gym.Env.
        """
        self._env.seed(seed)

    def step(self, action: Action) -> EnvTransition:
        """ Inherited from gym.Env.
        """
        return EnvTransition(*self._env.step(action))

    def step_and_render(self, action: Action, render: bool = False) -> EnvTransition:
        res = self.step(action)
        if render:
            self.render()
        return res

    def step_and_reset(self, action: Action) -> EnvTransition:
        transition = self.step(action)
        if transition.terminal:
            return EnvTransition(self.reset(), *transition[1:])
        else:
            return transition

    @property
    def unwrapped(self) -> gym.Env:
        """ Inherited from gym.Env.
        """
        return self._env.unwrapped

    @property
    def action_dim(self) -> int:
        """
        Extended method.
        Returns a ndim of action space.
        """
        return self._spec.action_dim

    @property
    def state_dim(self) -> Sequence[int]:
        """
        Extended method.
        Returns a shape of observation space.
        """
        return self._spec.state_dim

    @property
    def use_reward_monitor(self) -> bool:
        """ Atari wrappers need RewardMonitor for evaluation.
        """
        return self._spec.use_reward_monitor

    @property
    def observation_space(self) -> gym.Space:
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._env.action_space

    def extract(self, state: State) -> Array:
        """
        Extended method.
        Convert state to ndarray.
        It's useful for the cases where numpy.ndarray representation is too large to
        throw it to replay buffer directly.
        """
        return state  # type: ignore

    def save_history(self, file_name: str) -> None:
        """
        Extended method.
        Save agent's action history to file.
        """
        import warnings

        warnings.warn("This environment does not support save_history!")

    def __repr__(self) -> str:
        return "EnvExt({})".format(self._env)

    def as_cls(self, cls: Union[str, Type[Self]]) -> Optional[Self]:
        if isinstance(cls, str):
            return _as_class(self._env, lambda env: env.__class__.__name__ == cls)
        else:
            return _as_class(self._env, lambda env: isinstance(env, cls))


def _as_class(env: gym.Env, query: [[gym.Env], bool]) -> Optional[gym.Env]:
    if query(env):
        return env
    if not hasattr(env, "env"):
        return None
    parent = env.env
    return _as_class(parent, query)


def _use_reward_monitor(env: gym.Env) -> bool:
    if not hasattr(env, "env"):
        return False
    parent = env.env

    if isinstance(parent, RewardMonitor):
        return True
    else:
        return _use_reward_monitor(parent)
