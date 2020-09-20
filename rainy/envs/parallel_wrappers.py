from typing import Any, Iterable, Optional, Tuple, Type, Union

import numpy as np

from ..prelude import Action, Array, Self, State
from ..utils import RunningMeanStd
from .parallel import ParallelEnv, PEnvTransition


class ParallelEnvWrapper(ParallelEnv[Action, State]):
    def __init__(self, penv: ParallelEnv) -> None:
        self.penv = penv
        self.nworkers = penv.nworkers
        self._spec = self.penv._spec

    def close(self) -> None:
        self.penv.close()

    def reset(self) -> Array[State]:
        return self.penv.reset()

    def step(self, actions: Iterable[Action]) -> PEnvTransition:
        return self.penv.step(actions)

    def seed(self, seeds: Iterable[int]) -> None:
        self.penv.seed(seeds)

    def extract(self, states: Iterable[State]) -> Array:
        return self.penv.extract(states)

    def do_any(
        self,
        function: str,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
    ) -> Array[Any]:
        return self.penv.do_any(function, args, kwargs)

    def unwrapped(self) -> ParallelEnv:
        return self.penv

    def as_cls(self, cls: Union[str, Type[ParallelEnv]]) -> Optional[ParallelEnv]:
        if isinstance(cls, str):
            if self.__class__.__name__ == cls:
                return self
            else:
                return self.penv.as_cls(cls)
        elif isinstance(self, cls):
            return self
        else:
            return self.penv.as_cls(cls)


class FrameStackParallel(ParallelEnvWrapper):
    """Parallel version of atari_wrappers.FrameStack
    """

    def __init__(
        self, penv: ParallelEnv, nstack: int = 4, dtype: type = np.float32,
    ) -> None:
        super().__init__(penv)
        idx = 0
        shape = self.penv.state_dim
        for dim in shape:
            if dim == 1:
                idx += 1
            else:
                break
        self.shape = (nstack, *self.penv.state_dim[idx:])
        self.obs = np.zeros((self.nworkers, *self.shape), dtype=dtype)

    def step(self, actions: Iterable[Action]) -> PEnvTransition:
        transition = self.penv.step(actions)
        self.obs = np.roll(self.obs, shift=-1, axis=1)
        for i, _ in filter(lambda t: t[1], enumerate(transition.terminals)):
            self.obs[i] = 0.0
        self.obs[:, -1] = self.extract(transition.states).squeeze()
        return PEnvTransition(self.obs, *transition[1:])

    def reset(self) -> Array[State]:
        self.obs.fill(0)
        state = self.penv.reset()
        self.obs[:, -1] = self.extract(state).squeeze()
        return self.obs

    @property
    def state_dim(self) -> Tuple[int, ...]:
        return self.shape


class NormalizeObsParallel(ParallelEnvWrapper[Action, Array[float]]):
    def __init__(self, penv: ParallelEnv, obs_clip: float = 10.0) -> None:
        super().__init__(penv)
        self.obs_clip = obs_clip
        self._rms = RunningMeanStd(shape=self.state_dim)
        self._training_mode = True

    def step(self, actions: Iterable[Action]) -> PEnvTransition:
        transition = self.penv.step(actions)
        return PEnvTransition(self._filter_obs(transition.states), *transition[1:])

    def _filter_obs(self, obs: Array[Array[float]]) -> Array[Array[float]]:
        if self._training_mode:
            self._rms.update(obs)
        obs = np.clip(
            (obs - self._rms.mean) / self._rms.std(), -self.obs_clip, self.obs_clip
        )
        return obs

    def reset(self) -> Array[Array[float]]:
        obs = self.penv.reset()
        return self._filter_obs(obs)

    def copyto(self, other: Self) -> None:
        self.penv.copyto(other.penv)
        self._rms.copyto(other._rms)

    def set_mode(self, train: bool = False) -> None:
        self.penv.set_mode(train)
        self._training_mode = train


class NormalizeRewardParallel(ParallelEnvWrapper[Action, State]):
    def __init__(
        self, penv: ParallelEnv, reward_clip: float = 10.0, gamma: float = 0.99
    ) -> None:
        super().__init__(penv)
        self.reward_clip = reward_clip
        self.gamma = gamma
        self._rms = RunningMeanStd(shape=())
        self._ret = np.zeros(self.nworkers)
        self._training_mode = True

    def step(self, actions: Iterable[Action]) -> PEnvTransition:
        transition = self.penv.step(actions)
        if self._training_mode:
            ret = self._ret * self.gamma + transition.rewards
            self._rms.update(ret)
            self._ret = ret * (1.0 - transition.terminals)
        return transition.map_r(
            lambda r: np.clip(r / self._rms.std(), -self.reward_clip, self.reward_clip)
        )

    def reset(self) -> Array[State]:
        self._ret.fill(0.0)
        return self.penv.reset()

    def copyto(self, other: Self) -> None:
        self.penv.copyto(other.penv)
        self._rms.copyto(other._rms)

    def set_mode(self, train: bool = False) -> None:
        self.penv.set_mode(train)
        self._training_mode = train
