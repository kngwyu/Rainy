import pytest
from test_env import DummyEnv
import torch
from rainy.agents.aoc import AocRolloutStorage
from rainy.lib.rollout import RolloutSampler, RolloutStorage
from rainy.envs import DummyParallelEnv, MultiProcEnv, ParallelEnv
from rainy.net.policy import CategoricalDist
from rainy.net import recurrent
from rainy.utils import Device

NSTEP = 4
ACTION_DIM = 3


@pytest.mark.parametrize('penv', [
    DummyParallelEnv(lambda: DummyEnv(array_dim=(16, 16)), 6),
    MultiProcEnv(lambda: DummyEnv(array_dim=(16, 16)), 6)
])
def test_storage(penv: ParallelEnv) -> None:
    NWORKERS = penv.num_envs
    storage = RolloutStorage(NSTEP, penv.num_envs, Device())
    storage.set_initial_state(penv.reset())
    policy_dist = CategoricalDist(ACTION_DIM)
    for _ in range(NSTEP):
        state, reward, done, _ = penv.step([None] * NWORKERS)
        value = torch.rand(NWORKERS, dtype=torch.float32)
        policy = policy_dist(torch.rand(NWORKERS, ACTION_DIM))
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
    penv.close()


def test_oc_storage() -> None:
    penv = DummyParallelEnv(lambda: DummyEnv(array_dim=(16, 16)), 6)
    NWORKERS = penv.num_envs
    NOPTIONS = 4
    storage = AocRolloutStorage(NSTEP, penv.num_envs, Device(), NOPTIONS)
    storage.set_initial_state(penv.reset())
    policy_dist = CategoricalDist(ACTION_DIM)
    for _ in range(NSTEP):
        state, reward, done, _ = penv.step([None] * NWORKERS)
        value = torch.rand(NWORKERS, NOPTIONS)
        policy = policy_dist(torch.rand(NWORKERS, ACTION_DIM))
        storage.push(state, reward, done, value=value, policy=policy)
        options = torch.randint(NOPTIONS, (NWORKERS,), device=storage.device.unwrapped)
        is_new_options = torch.randint(2, (NWORKERS,), device=storage.device.unwrapped).byte()
        storage.push_options(options, is_new_options, 0.5)
    next_value = torch.randn(NWORKERS, NOPTIONS).max(dim=-1)[0]
    storage.calc_returns(next_value, 0.99, 0.01)
    assert tuple(storage.beta_adv.shape) == (NSTEP, NWORKERS)
    penv.close()


class TeState(recurrent.RnnState):
    def __init__(self, h):
        self.h = h

    def __getitem__(self, x):
        return TeState(self.h[x])

    def __setitem__(self, x, value):
        self.f[x] = value

    def fill_(self, f):
        self.h.fill_(f)

    def unsqueeze(self):
        return TeState(self.h.unsqueeze(0))


@pytest.mark.parametrize('penv, is_recurrent', [
    (DummyParallelEnv(lambda: DummyEnv(array_dim=(16, 16)), 6), False),
    (MultiProcEnv(lambda: DummyEnv(array_dim=(16, 16)), 6), False),
    (DummyParallelEnv(lambda: DummyEnv(array_dim=(16, 16)), 6), True),
    (DummyParallelEnv(lambda: DummyEnv(array_dim=(16, 16)), 8), True),
])
def test_sampler(penv: ParallelEnv, is_recurrent: bool) -> None:
    NWORKERS = penv.num_envs
    rnns = TeState(torch.arange(NWORKERS)) if is_recurrent else recurrent.DummyRnn.DUMMY_STATE
    storage = RolloutStorage(NSTEP, penv.num_envs, Device())
    storage.set_initial_state(penv.reset(), rnn_state=rnns)
    policy_dist = CategoricalDist(ACTION_DIM)
    for _ in range(NSTEP):
        state, reward, done, _ = penv.step([None] * NWORKERS)
        value = torch.rand(NWORKERS, dtype=torch.float32)
        policy = policy_dist(torch.rand(NWORKERS, ACTION_DIM))
        storage.push(state, reward, done, rnn_state=rnns, policy=policy, value=value)
    MINIBATCH = 12
    rnn_test = set()
    for batch in RolloutSampler(storage, penv, MINIBATCH):
        length = len(batch.states)
        assert length == MINIBATCH
        if isinstance(batch.rnn_init, TeState):
            assert batch.rnn_init.h.size(0) == MINIBATCH // NSTEP
            rnn_test.update(batch.rnn_init.h.cpu().tolist())
    if is_recurrent:
        assert len(rnn_test) > NWORKERS - (MINIBATCH // NSTEP)
    penv.close()
