from abc import ABC, abstractmethod
import multiprocessing as mp
from multiprocessing.connection import Connection
import numpy as np
from numpy import ndarray
from typing import Any, Callable, Generic, Iterable, Tuple
from ..util.meta import Array
from . import Action, EnvExt, State


class ParallelEnv(ABC, Generic[Action, State]):
    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def reset(self) -> Array[State]:
        pass

    @abstractmethod
    def step(
            self,
            actions: Iterable[Action]
    ) -> Tuple[Array[State], Array[float], Array[bool], Array[Any]]:
        pass

    @abstractmethod
    def seed(self, seed: int) -> None:
        pass

    @abstractmethod
    def num_envs(self) -> int:
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def state_dim(self) -> Tuple[int, ...]:
        pass

    @abstractmethod
    def states_to_array(self, states: Iterable[State]) -> Array:
        pass


def make_parallel_env(env_gen: Callable[[], EnvExt], nworkers: int) -> ParallelEnv:
    e = env_gen()
    if not isinstance(e, EnvExt):
        raise ValueError('Needs EnvExt, but given {}'.format(type(e)))
    if nworkers < 1:
        raise ValueError('nworkers must be larger than 0')
    elif nworkers == 1:
        return DummyParallelEnv(e, 1)
    else:
        return MultiProcEnv(env_gen, nworkers)


class MultiProcEnv(ParallelEnv):
    def __init__(self, env_gen: Callable[[], EnvExt], nworkers: int) -> None:
        assert nworkers >= 2
        self.envs = [_ProcHandler(env_gen()) for _ in range(nworkers)]
        self._reserved = env_gen()

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def reset(self) -> Array[State]:
        for env in self.envs:
            env.reset()
        return np.array([env.recv() for env in self.envs])

    def step(
            self,
            actions: Iterable[Action]
    ) -> Tuple[Array[State], Array[float], Array[bool], Array[Any]]:
        for env, action in zip(self.envs, actions):
            env.step(action)
        res = [env.recv() for env in self.envs]
        return tuple(map(np.array, zip(*res)))  # type: ignore

    def seed(self, seed: int) -> None:
        for env in self.envs:
            env.seed(seed)

    def num_envs(self) -> int:
        return len(self.envs)

    @property
    def action_dim(self) -> int:
        return self._reserved.action_dim

    @property
    def state_dim(self) -> Tuple[int, ...]:
        return self._reserved.state_dim

    def states_to_array(self, states: Iterable[State]) -> ndarray:
        return np.asarray([self._reserved.state_to_array(s) for s in states])


class _ProcHandler:
    def __init__(self, env: EnvExt) -> None:
        self.pipe, worker_pipe = mp.Pipe()
        self.worker = _ProcWorker(env, worker_pipe)
        self.worker.start()

    def close(self) -> None:
        self.pipe.send((_ProcWorker.CLOSE, None))

    def reset(self) -> None:
        self.pipe.send((_ProcWorker.RESET, None))

    def seed(self, seed: int) -> None:
        self.pipe.send((_ProcWorker.SEED, seed))

    def step(self, action: Action) -> None:
        self.pipe.send((_ProcWorker.STEP, action))

    def recv(self) -> Any:
        return self.pipe.recv()


class _ProcWorker(mp.Process):
    CLOSE = 0
    RESET = 1
    SEED = 2
    STEP = 3

    def __init__(self, env: EnvExt, pipe: Connection) -> None:
        super(_ProcWorker, self).__init__()
        self.env = env
        self.pipe = pipe

    def run(self):
        while True:
            op, arg = self.pipe.recv()
            if op == self.STEP:
                self.pipe.send(self.env.step_and_reset(arg))
            elif op == self.RESET:
                self.pipe.send(self.env.reset())
            elif op == self.SEED:
                self.env.seed(arg)
            elif op == self.CLOSE:
                self.env.close()
                self.pipe.close()
                break


class DummyParallelEnv(ParallelEnv):
    def __init__(self, gen: Callable[[], EnvExt], nworkers: int) -> None:
        self.envs = [gen() for _ in range(nworkers)]

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def reset(self) -> Array[State]:
        return np.array([e.reset() for e in self.envs])

    def step(
            self,
            actions: Iterable[Action]
    ) -> Tuple[Array[State], Array[float], Array[bool], Array[Any]]:
        res = [e.step_and_reset(a) for (a, e) in zip(actions, self.envs)]
        return tuple(map(np.array, zip(*res)))  # type: ignore

    def seed(self, seed: int) -> None:
        for env in self.envs:
            env.seed(seed)

    def num_envs(self) -> int:
        return len(self.envs)

    @property
    def action_dim(self) -> int:
        return self.envs[0].action_dim

    @property
    def state_dim(self) -> Tuple[int, ...]:
        return self.envs[0].state_dim

    def states_to_array(self, states: Iterable[State]) -> ndarray:
        return np.asarray([e.state_to_array(s) for (s, e) in zip(states, self.envs)])


class ParallelEnvWrapper(ParallelEnv):
    def __init__(self, penv: ParallelEnv) -> None:
        self.penv = penv

    def close(self) -> None:
        self.penv.close()

    def reset(self) -> Array[State]:
        return self.penv.reset()

    def step(
            self,
            actions: Iterable[Action]
    ) -> Tuple[Array[State], Array[float], Array[bool], Array[Any]]:
        return self.penv.step(actions)

    def seed(self, seed: int) -> None:
        self.penv.seed(seed)

    def num_envs(self) -> int:
        return self.penv.num_envs()

    @property
    def action_dim(self) -> int:
        return self.penv.action_dim

    @property
    def state_dim(self) -> Tuple[int, ...]:
        return self.penv.state_dim

    def states_to_array(self, states: Iterable[State]) -> ndarray:
        return self.penv.states_to_array(states)


class FrameStackParallel(ParallelEnvWrapper):
    def __init__(self, penv: ParallelEnv, nstack: int = 4, dtype: type = np.float32) -> None:
        super().__init__(penv)
        idx = 0
        shape = self.penv.state_dim
        for dim in shape:
            if dim == 1:
                idx += 1
            else:
                break
        self.shape = (nstack, *self.penv.state_dim[idx:])
        self.obs = np.zeros((self.num_envs(), *self.shape), dtype=dtype)

    def step(
            self,
            actions: Iterable[Action]
    ) -> Tuple[ndarray, Array[float], Array[bool], Array[Any]]:
        state, reward, done, info = self.penv.step(actions)
        self.obs = np.roll(self.obs, shift=-1, axis=1)
        for i, _ in filter(lambda t: t[1], enumerate(done)):
            self.obs[i] = 0.0
        self.obs[:, -1] = self.states_to_array(state).squeeze()
        return (self.obs, reward, done, info)

    def reset(self) -> Array[State]:
        self.obs.fill(0)
        state = self.penv.reset()
        self.obs[:, -1] = self.states_to_array(state).squeeze()
        return self.obs

    @property
    def state_dim(self) -> Tuple[int, ...]:
        return self.shape
