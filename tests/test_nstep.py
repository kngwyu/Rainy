import pytest
from test_env import DummyEnv
import torch
from rainy.agent.nstep_common import FeedForwardSampler, RolloutStorage
from rainy.envs import DummyParallelEnv, MultiProcEnv, ParallelEnv
from rainy.net.policy import softmax
from rainy.util import Device


@pytest.mark.parametrize('penv', [
    DummyParallelEnv(lambda: DummyEnv(array_dim=(16, 16)), 6),
    MultiProcEnv(lambda: DummyEnv(array_dim=(16, 16)), 6)
])
def test_storage(penv: ParallelEnv) -> None:
    NSTEP = 4
    ACTION_DIM = 3
    NWORKERS = penv.num_envs()
    states = penv.reset()
    storage = RolloutStorage(NSTEP, NWORKERS, Device())
    storage.set_initial_state(states)
    for _ in range(NSTEP):
        state, reward, done, _ = penv.step([None] * NWORKERS)
        value = torch.rand((NWORKERS, ACTION_DIM))
        policy = softmax(torch.rand(NWORKERS, ACTION_DIM))
        storage.push(state, reward, done, value=value, policy=policy)
    batch = storage.batch_states(penv)
    batch_shape = torch.Size((NSTEP * NWORKERS,))
    assert batch.shape == torch.Size((*batch_shape, 16, 16))
    sampler = FeedForwardSampler(storage, penv, 10)
    assert sampler.actions.shape == batch_shape
    assert sampler.returns.shape == batch_shape
    assert sampler.masks.shape == batch_shape
    assert sampler.values.shape == torch.Size((*batch_shape, ACTION_DIM))
    assert sampler.old_log_probs.shape == batch_shape
    assert sampler.rewards.shape == batch_shape
    penv.close()


@pytest.mark.parametrize('penv', [
    DummyParallelEnv(lambda: DummyEnv(array_dim=(16, 16)), 6),
    MultiProcEnv(lambda: DummyEnv(array_dim=(16, 16)), 6)
])
def test_sampler(penv: ParallelEnv) -> None:
    NSTEP = 4
    ACTION_DIM = 3
    NWORKERS = penv.num_envs()
    states = penv.reset()
    storage = RolloutStorage(NSTEP, NWORKERS, Device())
    storage.set_initial_state(states)
    for _ in range(NSTEP):
        state, reward, done, _ = penv.step([None] * NWORKERS)
        value = torch.rand((NWORKERS, ACTION_DIM))
        policy = softmax(torch.rand(NWORKERS, ACTION_DIM))
        storage.push(state, reward, done, policy=policy, value=value)
    MINIBATCH = 10
    for batch in FeedForwardSampler(storage, penv, MINIBATCH):
        length = len(batch.states)
        assert length == MINIBATCH or length == (NSTEP * NWORKERS) % MINIBATCH
    penv.close()
