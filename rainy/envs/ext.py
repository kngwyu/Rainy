from abc import ABC
import gym
from gym import spaces
from numpy import ndarray
from typing import Any, Generic, NamedTuple, Tuple
from ..prelude import Action, State


class EnvSpec(NamedTuple):
    """Properties which are common both in EnvExt and ParallelEnv
    """
    state_dim: Tuple[int, ...]
    action_dim: int
    action_space: gym.Space
    use_reward_monitor: bool

    def random_action(self) -> Action:
        return self.action_space.sample()

    def is_discrete(self) -> bool:
        return isinstance(self.action_space, spaces.Discrete)


class EnvExt(gym.Env, ABC, Generic[Action, State]):
    def __init__(self, env: gym.Env) -> None:
        self._env = env
        self.spec = EnvSpec(
            self._env.observation_space.shape,
            self._env.action_space.n,
            self._env.action_space,
            False
        )

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
        return self._env.step(action)

    def step_and_reset(self, action: Action) -> Tuple[State, float, bool, Any]:
        state, reward, done, info = self.step(action)
        if done:
            state = self.reset()
        return state, reward, done, info

    @property
    def unwrapped(self) -> gym.Env:
        """
        Inherited from gym.Env.
        """
        return self._env.unwrapped

    @property
    def action_dim(self) -> int:
        """
        Extended method.
        Returns a ndim of action space.
        """
        return self.spec.action_dim

    @property
    def state_dim(self) -> Tuple[int, ...]:
        """
        Extended method.
        Returns a shape of observation space.
        """
        return self.spec.state_dim

    @property
    def use_reward_monitor(self) -> bool:
        """Atari wrappers need RewardMonitor for evaluation.
        """
        return self.spec.use_reward_monitor

    @property
    def observation_space(self) -> gym.Space:
        return self._env.observation_space

    @property
    def action_space(self) -> gym.Space:
        return self._env.action_space

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

    def __repr__(self) -> str:
        return 'EnvExt({})'.format(self._env)
