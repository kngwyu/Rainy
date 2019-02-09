from enum import Enum
from functools import reduce
import numpy as np
from numpy.testing import assert_array_almost_equal
from rainy import envs
from rainy.envs import DummyParallelEnv, EnvExt,\
    EnvSpec, MultiProcEnv, ParallelEnv, FrameStackParallel
import pytest
from typing import Tuple


class State(Enum):
    START = 0
    ROAD1 = 1
    ROAD2 = 2
    DEATH = 3
    GOAL = 4

    def is_end(self) -> bool:
        return self == State.DEATH or self == State.GOAL

    def to_array(self, dim: tuple) -> np.array:
        length = reduce(lambda x, y: x * y, dim)
        return np.repeat(float(self.value) + 1.0, length).reshape(dim)


class DummyEnv(EnvExt):
    def __init__(self, array_dim: Tuple[int, ...] = (16, 16)) -> None:
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

    def reset(self) -> State:
        self.state = State.START
        return self.state

    def step(self, _action) -> Tuple[State, float, bool, dict]:
        prob = np.asarray(self.transition[self.state.value])
        self.state = State(np.random.choice(np.arange(5), 1, p=prob))
        return self.state, self.rewards[self.state.value], self.state.is_end(), {}

    @property
    def spec(self) -> EnvSpec:
        return EnvSpec(self.array_dim, 1, False)

    def close(self) -> None:
        pass

    def state_to_array(self, state: State) -> np.ndarray:
        return state.to_array(self.array_dim)


class DummyEnvDeterministic(DummyEnv):
    def __init__(self, array_dim: Tuple[int, ...] = (16, 16)) -> None:
        super().__init__(array_dim)
        self.transition = [
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]


@pytest.mark.parametrize('penv', [
    DummyParallelEnv(lambda: DummyEnv(), 4),
    MultiProcEnv(lambda: DummyEnv(), 4)
])
def test_penv(penv: ParallelEnv) -> None:
    states = penv.reset()
    for s in states:
        assert s == State.START
    ok = [False] * 4
    for _ in range(5):
        _, _, done, _ = penv.step(np.repeat(None, 4))
        for i, done in enumerate(done):
            ok[i] |= done
    for i in range(4):
        assert ok[i]
    assert penv.num_envs() == 4
    penv.close()


@pytest.mark.parametrize('penv, nstack', [
    (DummyParallelEnv(lambda: DummyEnvDeterministic(), 6), 4),
    (MultiProcEnv(lambda: DummyEnvDeterministic(), 6), 4),
    (DummyParallelEnv(lambda: DummyEnvDeterministic(), 6), 8)
])
def test_frame_stack(penv: ParallelEnv, nstack: int) -> None:
    penv = FrameStackParallel(penv, nstack=nstack)
    init = np.array(penv.reset())
    assert init.shape == (6, nstack, 16, 16)
    assert penv.state_dim == (nstack, 16, 16)
    for i in range(3):
        obs, *_ = penv.step([None] * penv.num_envs())
        if i < 2:
            assert_array_almost_equal(obs[:, -2 - i], init[:, -1])
    assert_array_almost_equal(obs[:, -2], np.zeros((6, 16, 16)))
    assert_array_almost_equal(obs[:, -1], np.ones((6, 16, 16)))
    assert_array_almost_equal(penv.reset(), obs)
    penv.close()


@pytest.mark.parametrize('style', ['dopamine', 'deepmind', 'baselines'])
def test_atari(style: str):
    atari = envs.Atari('Pong', style=style)
    STATE_DIM = (4, 84, 84)
    assert atari.state_dim == STATE_DIM
    assert atari.action_dim == 6
    s = atari.reset()
    assert s.shape == STATE_DIM
