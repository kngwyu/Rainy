from abc import ABC, abstractmethod
import multiprocessing as mp
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
import numpy as np
from numpy import ndarray
from torch import Tensor
from typing import Any, Callable, Generic, Iterable, List, Tuple, TypeVar

from . import EnvExt

Action = TypeVar('Action', int, Tensor)
State = TypeVar('State')


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


class ParallelEnv(ABC, Generic[Action, State]):
    @abstractmethod
    def reset(self) -> State:
        pass

    @abstractmethod
    def step_async(self, actions: Iterable[Action]) -> List[Tuple[State, float, bool, Any]]:
        pass

    @abstractmethod
    def step_sync(self, actions: Iterable[Action]) -> List[Tuple[State, float, bool, Any]]:
        pass

    def step(self, actions: Iterable[Action]) -> List[Tuple[State, float, bool, Any]]:
        return self.step_sync(actions)

    def states_to_array(self, states: State) -> ndarray:
        pass


class MultiProcEnv(ParallelEnv):
    def __init__(self, e: EnvExt, env_gen: Callable[[], EnvExt], num_workers: int) -> None:
        assert num_workers >= 2
        self.envs = [_ProcHandler(env_gen()) for _ in range(num_workers)]

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def reset(self) -> List[State]:
        return [e.reset() for e in self.envs]

    def step_async(self, actions: Iterable[Action]) -> List[Tuple[State, float, bool, Any]]:
        pass

    def step_sync(self, actions: Iterable[Action]) -> List[Tuple[State, float, bool, Any]]:
        pass

    def seed(self, seed: int) -> None:
        pass


class _ProcHandler:
    def __init__(self, env: EnvExt) -> None:
        self.pipe, worker_pipe = mp.Pipe()
        self.worker = _ProcWorker(env, worker_pipe)
        self.worker.start()

    def close(self) -> None:
        self.pipe.send((_ProcWorker.CLOSE, None))

    def reset(self) -> State:
        self.pipe.send((_ProcWorker.RESET, None))
        return self.pipe.recv()

    def seed(self, seed: int) -> None:
        self.pipe.send((_ProcWorker.SEED, seed))

    def step(self, action: Action) -> State:
        self.pipe.send((_ProcWorker.STEP, action))
        return self.pipe.recv()


class _ProcWorker(Process):
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

    def reset(self) -> List[State]:
        return [e.reset() for e in self.envs]

    def step_sync(self, actions: Iterable[Action]) -> List[Tuple[State, float, bool, Any]]:
        return [e.reset() for e in self.envs]

    def step_async(self, actions: Iterable[Action]) -> List[Tuple[State, float, bool, Any]]:
        pass

    @property
    def action_dim(self) -> int:
        return self.envs[0].action_dim

    @property
    def state_dims(self) -> Tuple[int]:
        return self.envs[0].state_dims

    def states_to_array(self, states: List[State]) -> ndarray:
        return np.asarray([e.state_to_array(s) for e, s in zip(self.envs, states)])
