from abc import ABC, abstractmethod
from torch import multiprocessing as mp
from typing import Any

from ..envs import EnvExt
from ..prelude import Action, State


class Actor(ABC, Generic[Action, State]):
    def close(self) -> None:
        pass

    @abstractmethod
    def reset(self, initial_state: State) -> None:
        pass

    @abstractmethod
    def sync(self, other: Any):
        pass


class _ActorProcess(mp.Process):
    STEP = 0
    RESET = 1
    N_STEPS = 2
    SYNC = 3
    CLOSE = 4

    def __init__(self, actor: Actor, env: EnvExt, pipe: mp.Connection) -> None:
        super().__init__()
        self.actor = actor
        self.env = env
        self.pipe = pipe

    def run(self):
        while True:
            op, arg = self.pipe.recv()
            if op == self.STEP:
                self.pipe.send(self.env.step)
            elif op == self.RESET:
                self.pipe.send(self.env.reset)
            elif op == self.N_STEPS:
                pass
            elif op == self.SYNC:
                self.actor.sync(arg)
            elif op == self.CLOSE:
                self.actor.close()
                self.env.close()
                self.pipe.close()
                break
            else:
                raise NotImplementedError("Not-supported operation: {}".format(op))


