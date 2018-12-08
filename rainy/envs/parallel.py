from abc import ABC, abstractmethod
import multiprocessing as mp
from multiprocessing.connection import Connection
import numpy as np
from numpy import ndarray
from torch import Tensor
from typing import Any, Callable, Generic, Iterable, List, Tuple, TypeVar

from . import EnvExt

Action = TypeVar('Action', int, Tensor)
State = TypeVar('State')


class ParallelEnv(ABC, Generic[Action, State]):
    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def reset(self) -> List[State]:
        pass

    @abstractmethod
    def step(self, actions: Iterable[Action]) -> List[Tuple[State, float, bool, Any]]:
        pass

    def states_to_array(self, states: Iterable[State]) -> ndarray:
        return np.asarray([s for s in states])


def make_parallel_env(env_gen: Callable[[], EnvExt], num_workers: int) -> ParallelEnv:
    e = env_gen()
    if not isinstance(e, EnvExt):
        raise ValueError('Needs EnvExt, but given {}'.format(type(e)))
    if num_workers < 1:
        raise ValueError('num_workers must be larger than 0')
    elif num_workers == 1:
        return DummyParallelEnv(e, 1)
    else:
        return MultiProcEnv(e, env_gen, num_workers)


class MultiProcEnv(ParallelEnv):
    def __init__(self, e: EnvExt, env_gen: Callable[[], EnvExt], num_workers: int) -> None:
        assert num_workers >= 2
        self.envs = [_ProcHandler(env_gen()) for _ in range(num_workers)]

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def reset(self) -> List[State]:
        for env in self.envs():
            env.reset()
        return [env.recv() for env in self.envs]

    def step(self, actions: Iterable[Action]) -> List[Tuple[State, float, bool, Any]]:
        for env, action in zip(self.envs, actions):
            env.step(action)
        return [env.recv() for env in self.envs]


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
        self.env = env
        self.pipe = pipe

    def run(self):
        while True:
            op, arg = self.pipe.recv()
            if op == self.STEP:
                self.pipe.send(self.env.step(arg))
            elif op == self.RESET:
                self.pipe.send(self.env.reset(arg))
            elif op == self.SEED:
                self.env.seed(arg)
            elif op == self.CLOSE:
                self.env.close()
                self.pipe.close()
                break


class DummyParallelEnv(ParallelEnv):
    def __init__(self, gen: Callable[[], EnvExt], num_workers: int) -> None:
        self.envs = [gen() for _ in range(num_workers)]

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def reset(self) -> List[State]:
        return [e.reset() for e in self.envs]

    def step(self, actions: Iterable[Action]) -> List[Tuple[State, float, bool, Any]]:
        return [e.step() for e in self.envs]
