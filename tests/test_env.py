import numpy as np
from numpy.testing import assert_array_almost_equal
from rainy import envs
from rainy.envs import DummyParallelEnv, MultiProcEnv, ParallelEnv, FrameStackParallel
from rainy.envs.testing import DummyEnv, DummyEnvDeterministic, State
import pytest


@pytest.mark.parametrize(
    "penv",
    [DummyParallelEnv(lambda: DummyEnv(), 4), MultiProcEnv(lambda: DummyEnv(), 4)],
)
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
    assert penv.num_envs == 4
    penv.close()


@pytest.mark.parametrize(
    "penv, nstack",
    [
        (DummyParallelEnv(lambda: DummyEnvDeterministic(), 6), 4),
        (MultiProcEnv(lambda: DummyEnvDeterministic(), 6), 4),
        (DummyParallelEnv(lambda: DummyEnvDeterministic(), 6), 8),
    ],
)
def test_frame_stack(penv: ParallelEnv, nstack: int) -> None:
    penv = FrameStackParallel(penv, nstack=nstack)
    penv.seed(np.arange(penv.num_envs))
    init = np.array(penv.reset())
    assert init.shape == (6, nstack, 16, 16)
    assert penv.state_dim == (nstack, 16, 16)
    for i in range(3):
        obs, *_ = penv.step([None] * penv.num_envs)
        if i < 2:
            assert_array_almost_equal(obs[:, -2 - i], init[:, -1])
    assert_array_almost_equal(obs[:, -2], np.zeros((6, 16, 16)))
    assert_array_almost_equal(obs[:, -1], np.ones((6, 16, 16)))
    assert_array_almost_equal(penv.reset(), obs)
    penv.close()


@pytest.mark.parametrize("style", ["dopamine", "deepmind", "baselines"])
def test_atari(style: str):
    atari = envs.Atari("Pong", style=style)
    assert atari.spec.use_reward_monitor
    STATE_DIM = (4, 84, 84)
    assert atari.state_dim == STATE_DIM
    assert atari.action_dim == 6
    s = atari.reset()
    assert s.shape == STATE_DIM
