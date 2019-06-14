"""Dummy environment for testing
"""
from enum import Enum
from gym.spaces import Discrete
import numpy as np
from typing import Tuple
from .ext import EnvExt, EnvSpec

ACTION_DIM = 10


class State(Enum):
    START = 0
    ROAD1 = 1
    ROAD2 = 2
    DEATH = 3
    GOAL = 4

    def is_end(self) -> bool:
        return self == State.DEATH or self == State.GOAL

    def to_array(self, dim: tuple) -> np.array:
        return np.repeat(float(self.value) + 1.0, np.prod(dim)).reshape(dim)


class DummyEnv(EnvExt):
    def __init__(self, array_dim: Tuple[int, ...] = (16, 16), flatten: bool = False) -> None:
        self.state = State.START
        self.transition = [
            [0.0, 0.7, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.4, 0.6],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
        self.rewards = [0., 0., 0., -10., 20.]
        self.array_dim = array_dim
        self.flatten = flatten

    def reset(self) -> State:
        self.state = State.START
        return self.state

    def step(self, _action) -> Tuple[State, float, bool, dict]:
        prob = np.asarray(self.transition[self.state.value])
        self.state = State(np.random.choice(np.arange(5), 1, p=prob))
        return self.state, self.rewards[self.state.value], self.state.is_end(), {}

    @property
    def spec(self) -> EnvSpec:
        if self.flatten:
            dim = (np.prod(self.array_dim),)
        else:
            dim = self.array_dim
        return EnvSpec(dim, Discrete(ACTION_DIM))

    def seed(self, int) -> None:
        pass

    def close(self) -> None:
        pass

    def extract(self, state: State) -> np.ndarray:
        res = state.to_array(self.array_dim)
        if self.flatten:
            return res.flatten()
        else:
            return res


class DummyEnvDeterministic(DummyEnv):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.transition = [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
