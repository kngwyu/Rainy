import multiprocessing as mp
from abc import ABC, abstractmethod
from multiprocessing.connection import Connection
from typing import (
    Any,
    Callable,
    Generic,
    Iterable,
    NamedTuple,
    Optional,
    Sequence,
    Union,
    Type,
)

import numpy as np
from numpy import ndarray

from ..prelude import Action, Array, GenericNamedMeta, Self, State
from ..utils import mp_utils
from .ext import EnvExt, EnvSpec

EnvGen = Callable[[], EnvExt]


class PEnvTransition(NamedTuple, Generic[State], metaclass=GenericNamedMeta):
    states: Array[State]
    rewards: Array[float]
    terminals: Array[bool]
    infos: Array[dict]

    def map_r(self, f: Callable[[Array[float]], Array[float]]) -> Self:
        s, r, t, i = self
        return PEnvTransition(s, f(r), t, i)


class ParallelEnv(ABC, Generic[Action, State]):
    nworkers: int
    spec: EnvSpec

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def reset(self) -> Array[State]:
        pass

    @abstractmethod
    def step(self, actions: Iterable[Action]) -> PEnvTransition:
        pass

    @abstractmethod
    def seed(self, seeds: Iterable[int]) -> None:
        pass

    @property
    def action_dim(self) -> int:
        return self._spec.action_dim

    @property
    def state_dim(self) -> Sequence[int]:
        return self._spec.state_dim

    @property
    def use_reward_monitor(self) -> bool:
        return self._spec.use_reward_monitor

    @abstractmethod
    def extract(self, states: Iterable[State]) -> Array:
        """
        Convert Sequence of states to ndarray.
        """
        pass

    @abstractmethod
    def do_any(
        self,
        function: str,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
    ) -> Array[Any]:
        """
        Execute any function.
        """
        pass

    def copyto(self, other: Self) -> None:
        pass

    def set_mode(self, train: bool = False) -> None:
        pass

    def unwrapped(self) -> Self:
        return self

    def as_cls(self, cls: Union[str, Type[Self]]) -> Optional[Self]:
        """Get the specified class from the wrapper `self`
        """
        if isinstance(cls, str):
            if self.__class__.__name__ == cls:
                return self
            else:
                return None
        elif isinstance(self, cls):
            return self
        else:
            return None


class MultiProcEnv(ParallelEnv):
    def __init__(self, env_gen: EnvGen, nworkers: int) -> None:
        assert nworkers >= 2
        envs = [env_gen() for _ in range(nworkers)]
        self.to_array = envs[0].extract
        self._spec = envs[0]._spec
        self.envs = [_ProcHandler(i, envs[i]) for i in range(nworkers)]
        self.nworkers = nworkers

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def reset(self) -> Array[State]:
        for env in self.envs:
            env.reset()
        return np.array([env.recv() for env in self.envs])

    def step(self, actions: Iterable[Action]) -> PEnvTransition:
        for env, action in zip(self.envs, actions):
            env.step(action)
        res = [env.recv() for env in self.envs]
        return PEnvTransition(*map(np.array, zip(*res)))

    def seed(self, seeds: Iterable[int]) -> None:
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def extract(self, states: Iterable[State]) -> ndarray:
        return np.asarray([self.to_array(s) for s in states])

    def do_any(
        self,
        function: str,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
    ) -> Array[Any]:
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        for env in self.envs:
            env.do_any(function, args, kwargs)
        return np.array([env.recv() for env in self.envs])


class _ProcHandler:
    def __init__(self, envid: int, env: EnvExt) -> None:
        self.pipe, worker_pipe = mp.Pipe()
        self.worker = _ProcWorker(envid, env, worker_pipe)
        self.worker.start()

    def close(self) -> None:
        self.pipe.send((_ProcWorker.CLOSE, None))

    def reset(self) -> None:
        self.pipe.send((_ProcWorker.RESET, None))

    def seed(self, seed: int) -> None:
        self.pipe.send((_ProcWorker.SEED, seed))

    def step(self, action: Action) -> None:
        self.pipe.send((_ProcWorker.STEP, action))

    def do_any(self, function: str, args: tuple, kwargs: dict) -> None:
        self.pipe.send((_ProcWorker.ANY, (function, args, kwargs)))

    def recv(self) -> Any:
        return self.pipe.recv()


class _ProcWorker(mp.Process):
    CLOSE = 0
    RESET = 1
    SEED = 2
    STEP = 3
    ANY = 4

    def __init__(self, envid: int, env: EnvExt, pipe: Connection) -> None:
        super().__init__()
        self.envid = envid
        self.env = env
        self.pipe = pipe

    def run(self):
        def _loop():
            while True:
                op, arg = self.pipe.recv()
                if op == self.STEP:
                    self.pipe.send(self.env.step_and_reset(arg))
                elif op == self.ANY:
                    fname, args, kwargs = arg
                    fn = getattr(self.env.unwrapped, fname)
                    self.pipe.send(fn(*args, **kwargs))
                elif op == self.RESET:
                    self.pipe.send(self.env.reset())
                elif op == self.SEED:
                    self.env.seed(arg)
                elif op == self.CLOSE:
                    self.env.close()
                    self.pipe.close()
                    break
                else:
                    raise NotImplementedError("Not-supported operation: {}".format(op))

        mp_utils.pretty_loop(self.envid, _loop)


class DummyParallelEnv(ParallelEnv):
    def __init__(self, env_gen: EnvGen, nworkers: int) -> None:
        self.envs = [env_gen() for _ in range(nworkers)]
        self._spec = self.envs[0]._spec
        self.nworkers = nworkers

    def close(self) -> None:
        for env in self.envs:
            env.close()

    def reset(self) -> Array[State]:
        return np.array([e.reset() for e in self.envs])

    def step(self, actions: Iterable[Action]) -> PEnvTransition:
        res = [e.step_and_reset(a) for (a, e) in zip(actions, self.envs)]
        return PEnvTransition(*map(np.array, zip(*res)))

    def seed(self, seeds: Iterable[int]) -> None:
        for env, seed in zip(self.envs, seeds):
            env.seed(seed)

    def extract(self, states: Iterable[State]) -> ndarray:
        return np.asarray([e.extract(s) for (s, e) in zip(states, self.envs)])

    def do_any(
        self,
        function: str,
        args: Optional[tuple] = None,
        kwargs: Optional[dict] = None,
    ) -> Array[Any]:
        if args is None:
            args = ()
        if kwargs is None:
            kwargs = {}
        res = []
        for env in self.envs:
            fn = getattr(env.unwrapped, function)
            res.append(fn(*args, **kwargs))
        return np.array(res)
