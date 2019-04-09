import pytest
from test_env import DummyEnv
import torch
from rainy.lib.rollout import RolloutSampler, RolloutStorage
from rainy.envs import DummyParallelEnv, MultiProcEnv, ParallelEnv
from rainy.net.policy import CategoricalHead
from rainy.net import recurrent
from rainy.utils import Device


@pytest.mark.parametrize('penv', [
    DummyParallelEnv(lambda: DummyEnv(array_dim=(16, 16)), 6),
    MultiProcEnv(lambda: DummyEnv(array_dim=(16, 16)), 6)
])
def test_storage(penv: ParallelEnv) -> None:
    NSTEP = 4
    ACTION_DIM = 3
    NWORKERS = penv.num_envs
    states = penv.reset()
    storage = RolloutStorage(NSTEP, NWORKERS, Device())
    storage.set_initial_state(states)
    policy_head = CategoricalHead(ACTION_DIM)
    for _ in range(NSTEP):
        state, reward, done, _ = penv.step([None] * NWORKERS)
        value = torch.rand(NWORKERS, dtype=torch.float32)
        policy = policy_head(torch.rand(NWORKERS, ACTION_DIM))
        storage.push(state, reward, done, value=value, policy=policy)
    batch = storage.batch_states(penv)
    batch_shape = torch.Size((NSTEP * NWORKERS,))
    assert batch.shape == torch.Size((*batch_shape, 16, 16))
    sampler = RolloutSampler(storage, penv, 10)
    assert sampler.actions.shape == batch_shape
    assert sampler.returns.shape == batch_shape
    assert sampler.masks.shape == batch_shape
    assert sampler.values.shape == batch_shape
    assert sampler.old_log_probs.shape == batch_shape
    assert sampler.rewards.shape == batch_shape
    penv.close()


@pytest.mark.parametrize('penv, is_recurrent', [
    (DummyParallelEnv(lambda: DummyEnv(array_dim=(16, 16)), 6), False),
    (MultiProcEnv(lambda: DummyEnv(array_dim=(16, 16)), 6), False),
    (DummyParallelEnv(lambda: DummyEnv(array_dim=(16, 16)), 6), True),
    (DummyParallelEnv(lambda: DummyEnv(array_dim=(16, 16)), 8), True),
])
def test_sampler(penv: ParallelEnv, is_recurrent: bool) -> None:
    NSTEP = 4
    ACTION_DIM = 3
    NWORKERS = penv.num_envs
    states = penv.reset()
    rnns = recurrent.GruState(torch.zeros((NWORKERS, 10))) \
        if is_recurrent else recurrent.DummyRnn.DUMMY_STATE
    storage = RolloutStorage(NSTEP, NWORKERS, Device())
    storage.set_initial_state(states, rnn_state=rnns)
    policy_head = CategoricalHead(ACTION_DIM)
    for _ in range(NSTEP):
        state, reward, done, _ = penv.step([None] * NWORKERS)
        value = torch.rand(NWORKERS, dtype=torch.float32)
        policy = policy_head(torch.rand(NWORKERS, ACTION_DIM))
        storage.push(state, reward, done, rnn_state=rnns, policy=policy, value=value)
    MINIBATCH = 12
    for batch in RolloutSampler(storage, penv, MINIBATCH):
        length = len(batch.states)
        assert length == MINIBATCH or length == (NSTEP * NWORKERS) % MINIBATCH
        if isinstance(batch.rnn_init, recurrent.GruState):
            assert batch.rnn_init.h.size(0) == MINIBATCH // NSTEP
    penv.close()
