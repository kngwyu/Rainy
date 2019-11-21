from abc import ABC, abstractmethod
import multiprocessing as mp
from multiprocessing.connection import Connection
import numpy as np
from numpy import ndarray
from typing import Any, Callable, Generic, Iterable, Sequence, Tuple
from .ext import EnvExt, EnvSpec
from ..prelude import Action, Array, State

EnvGen = Callable[[], EnvExt]


class ParallelEnv(ABC, Generic[Action, State]):
    num_envs: int
    spec: EnvSpec

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def reset(self) -> Array[State]:
        pass

    @abstractmethod
    def step(
        self, actions: Iterable[Action]
    ) -> Tuple[Array[State], Array[float], Array[bool], Array[Any]]:
        pass

    @abstractmethod
    def seed(self, seeds: Iterable[int]) -> None:
        pass

    @property
    def action_dim(self) -> int:
        return self.spec.action_dim

    @property
    def state_dim(self) -> Sequence[int]:
        return self.spec.state_dim

    @property
    def use_reward_monitor(self) -> bool:
        return self.spec.use_reward_monitor

    @abstractmethod
    def extract(self, states: Iterable[State]) -> Array:
        """
        Convert Sequence of states to ndarray.
        """
        pass


class MultiProcEnv(ParallelEnv):
    def __init__(self, env_gen: EnvGen, nworkers: int) -> None:
        assert nworkers >= 2
        envs_tmp = [env_gen() for _ in range(nworkers)]
        self.to_array = envs_tmp[0].extract
        self.spec = envs_tmp[0].spec
        self.envs = [_ProcHandler(e) for e in envs_tmp]
        self.num_envs = nworkers

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def reset(self) -> Array[State]:
        for env in self.envs:
            env.reset()
        return np.array([env.recv() for env in self.envs])

    def step(
        self, actions: Iterable[Action]
    ) -> Tuple[Array[State], Array[float], Array[bool], Array[Any]]:
        for env, action in zip(self.envs, actions):
            env.step(action)
        res = [env.recv() for env in self.envs]
        return tuple(map(np.array, zip(*res)))  # type: ignore

    def seed(self, seeds: Iterable[int]) -> None:
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def extract(self, states: Iterable[State]) -> ndarray:
        return np.asarray([self.to_array(s) for s in states])


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
    def __init__(self, env_gen: EnvGen, nworkers: int) -> None:
        self.envs = [env_gen() for _ in range(nworkers)]
        self.spec = self.envs[0].spec
        self.num_envs = nworkers

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def reset(self) -> Array[State]:
        return np.array([e.reset() for e in self.envs])

    def step(
        self, actions: Iterable[Action]
    ) -> Tuple[Array[State], Array[float], Array[bool], Array[Any]]:
        res = [e.step_and_reset(a) for (a, e) in zip(actions, self.envs)]
        return tuple(map(np.array, zip(*res)))  # type: ignore

    def seed(self, seeds: Iterable[int]) -> None:
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def extract(self, states: Iterable[State]) -> ndarray:
        return np.asarray([e.extract(s) for (s, e) in zip(states, self.envs)])
