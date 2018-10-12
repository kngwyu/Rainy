from abc import ABC, abstractmethod
from concurrent import futures
from multiprocessing import Pipe, Process
from multiprocessing.connection import Connection
from typing import Any, Callable, List, Tuple

from . import Action, EnvExt, EnvGen, State


def crate_parallel_env(env_gen: EnvGen, num_workers: int) -> ParallelEnv:
    e = env_gen()
    if not isinstance(e, EnvExt):
        raise ValueError('Needs EnvExt, but given {}'.format(type(e)))
    if num_workers < 1:
        raise ValueError('num_workers must be larger than 2')
    elif num_workes == 1:
        return DummyParallelEnv(e)
    else:
        return ParallelEnv(e, env_gen, num_workers)


class ParallelEnv(EnvExt):
    def __init__(self, e: EnvExt, env_gen: EnvGen, num_workers: int) -> None:
        assert num_workers >= 2
        super().__init__(e)
        self.envs = [_ProcessHandler(env_gen()) for _ in range(num_workers)]

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def reset(self) -> List[State]:
        for env in self.envs:
            env.reset()

    def seed(self, seed: int) -> None:
        pass


class _ProcessHandler:
    def __init__(self, env: EnvExt) -> None:
        self.pipe, worker_pipe = mp.Pipe()
        self.worker = _ProcessWorker(env)
        self.worker.start()

    def close(self) -> None:
        self.pipe.send((_ProcessWorker.CLOSE, seed))

    def reset(self) -> State:
        self.pipe.send((_ProcessWorker.RESET, None))
        return self.pipe.recv()

    def seed(self, seed: int) -> None:
        self.pipe.send((_ProcessWorker.SEED, seed))

    def step(self, action: Action) -> State:
        self.pipe.send((_ProcessWorker.STEP, action))
        return self.pipe.recv()


class _ProcessWorker(mp.Process):
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
    def __init__(self, env: EnvExt) -> None:
        super(EnvExt, self).__init__(env)

    def reset(self) -> List[State]:
        return [self._env.reset()]

    def step(self, action: Action) -> List[Tuple[State, float, bool, Any]]:
        return [self._env.step(action)]



