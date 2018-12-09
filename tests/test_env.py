from enum import Enum
import numpy as np
from rainy.envs import DummyParallelEnv, EnvExt, MultiProcEnv, ParallelEnv
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

    def to_numpy(self) -> np.array:
        return np.repeat(self.value, 16).reshape(4, 4)


class DummyEnv(EnvExt):
    def __init__(self) -> None:
        self.state = State.START
        self.transition = [
            [0.0, 0.7, 0.3, 0.0, 0.0],
            [0.0, 0.0, 0.8, 0.2, 0.0],
            [0.0, 0.0, 0.0, 0.4, 0.6],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0],
        ]
        self.rewards = [0., 0., 0., -10., 20.]

    def reset(self) -> np.array:
        self.state = State.START
        return self.state.to_numpy()

    def step(self, _action) -> Tuple[np.array, float, bool, None]:
        prob = np.asarray(self.transition[self.state.value])
        self.state = State(np.random.choice(np.arange(5), 1, p=prob))
        return self.state.to_numpy(), self.rewards, self.state.is_end(), None

    def action_dim(self) -> int:
        return 1

    def state_dim(self) -> Tuple[int, int]:
        return (4, 4)

    def close(self) -> None:
        pass


@pytest.mark.parametrize('penv', [
    DummyParallelEnv(lambda: DummyEnv(), 4),
    MultiProcEnv(lambda: DummyEnv(), 4)
])
def test_penv(penv: ParallelEnv) -> None:
    states = penv.reset()
    for s in states:
        np.testing.assert_array_almost_equal(s, State.START.to_numpy())
    ok = [False] * 4
    for _ in range(5):
        _, _, done, _ = map(np.asarray, zip(*penv.step(np.repeat(None, 4))))
        for i, done in enumerate(done):
            ok[i] |= done
    for i in range(4):
        assert ok[i]
    assert penv.num_envs() == 4
    penv.close()

