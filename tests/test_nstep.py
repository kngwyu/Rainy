import pytest
from test_env import DummyEnv
import torch
from rainy.agent.nstep_common import RolloutStorage
from rainy.envs import DummyParallelEnv, MultiProcEnv, ParallelEnv
from rainy.util import Device


@pytest.mark.parametrize('penv', [
    DummyParallelEnv(lambda: DummyEnv(array_dim=(16, 16)), 6),
    MultiProcEnv(lambda: DummyEnv(array_dim=(16, 16)), 6)
])
def test_storage(penv: ParallelEnv) -> None:
    NSTEP = 4
    NWORKERS = penv.num_envs()
    states = penv.reset()
    storage = RolloutStorage(NSTEP, NWORKERS, Device())
    storage.set_initial_state(states)
    for _ in range(NSTEP):
        state, reward, done, _ = penv.step([None] * NWORKERS)
        storage.push(state, reward, done)
    batch = storage.batched_states(penv)
    assert batch.shape == torch.Size((NSTEP * NWORKERS, 16, 16))
    penv.close()
