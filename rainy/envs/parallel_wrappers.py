import numpy as np
from typing import Any, Iterable, Tuple
from .parallel import ParallelEnv
from ..prelude import Action, Array, State
from ..utils import RunningMeanStd


class ParallelEnvWrapper(ParallelEnv[Action, State]):
    def __init__(self, penv: ParallelEnv) -> None:
        self.penv = penv
        self.num_envs = penv.num_envs
        self.spec = self.penv.spec

    def close(self) -> None:
        self.penv.close()

    def reset(self) -> Array[State]:
        return self.penv.reset()

    def step(
        self, actions: Iterable[Action]
    ) -> Tuple[Array[State], Array[float], Array[bool], Array[Any]]:
        return self.penv.step(actions)

    def seed(self, seeds: Iterable[int]) -> None:
        self.penv.seed(seeds)

    def extract(self, states: Iterable[State]) -> Array:
        return self.penv.extract(states)


class FrameStackParallel(ParallelEnvWrapper):
    """Parallel version of atari_wrappers.FrameStack
    """

    def __init__(
        self, penv: ParallelEnv, nstack: int = 4, dtype: type = np.float32
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
        self.obs = np.zeros((self.num_envs, *self.shape), dtype=dtype)

    def step(
        self, actions: Iterable[Action]
    ) -> Tuple[Array, Array[float], Array[bool], Array[Any]]:
        state, reward, done, info = self.penv.step(actions)
        self.obs = np.roll(self.obs, shift=-1, axis=1)
        for i, _ in filter(lambda t: t[1], enumerate(done)):
            self.obs[i] = 0.0
        self.obs[:, -1] = self.extract(state).squeeze()
        return (self.obs, reward, done, info)

    def reset(self) -> Array[State]:
        self.obs.fill(0)
        state = self.penv.reset()
        self.obs[:, -1] = self.extract(state).squeeze()
        return self.obs

    @property
    def state_dim(self) -> Tuple[int, ...]:
        return self.shape


class NormalizeObs(ParallelEnvWrapper[Action, Array[float]]):
    def __init__(self, penv: ParallelEnv, obs_clip: float = 10.0) -> None:
        super().__init__(penv)
        self.obs_clip = obs_clip
        self._rms = RunningMeanStd(shape=self.state_dim)
        self.training_mode = True

    def step(
        self, actions: Iterable[Action]
    ) -> Tuple[Array[Array[float]], Array[float], Array[bool], Array[Any]]:
        state, reward, done, info = self.penv.step(actions)
        return self._filter_obs(state), reward, done, info

    def _filter_obs(self, obs: Array[Array[float]]) -> Array[Array[float]]:
        if self.training_mode:
            self._rms.update(obs)  # type: ignore
        obs = np.clip(
            (obs - self._rms.mean) / self._rms.std(), -self.obs_clip, self.obs_clip
        )
        return obs

    def reset(self) -> Array[Array[float]]:
        obs = self.penv.reset()
        return self._filter_obs(obs)


class NormalizeReward(ParallelEnvWrapper[Action, State]):
    def __init__(
        self, penv: ParallelEnv, reward_clip: float = 10.0, gamma: float = 0.99
    ) -> None:
        super().__init__(penv)
        self.reward_clip = reward_clip
        self.gamma = gamma
        self._rms = RunningMeanStd(shape=())
        self.ret = np.zeros(self.num_envs)

    def step(
        self, actions: Iterable[Action]
    ) -> Tuple[Array[State], Array[float], Array[bool], Array[Any]]:
        state, reward, done, info = self.penv.step(actions)
        self.ret = self.ret * self.gamma + reward
        self._rms.update(self.ret)
        reward = np.clip(reward / self._rms.std(), -self.reward_clip, self.reward_clip)
        self.ret[done] = 0.0
        return state, reward, done, info

    def reset(self) -> Array[State]:
        self.ret = np.zeros(self.num_envs)
        return self.penv.reset()
