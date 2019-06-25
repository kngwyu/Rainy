from functools import partial
import numpy as np
import pytest
import rainy
from rainy.agents import A2cAgent, A2ocAgent, PpoAgent
from rainy import net
from test_env import DummyEnvDeterministic


@pytest.mark.parametrize('n, make_ag', [(6, A2cAgent), (12, A2cAgent), (6, PpoAgent)])
def test_eval_parallel(n: int, make_ag: callable) -> None:
    c = rainy.Config()
    c.nworkers = 6
    c.nsteps = 5
    c.set_parallel_env(rainy.envs.MultiProcEnv)
    c.set_net_fn('actor-critic', net.actor_critic.fc_shared(units=[32, 32]))
    c.set_net_fn('option-critic', net.option_critic.fc_shared(units=[32, 32]))
    c.set_env(partial(DummyEnvDeterministic, flatten=True))
    ag = make_ag(c)
    entropy = np.zeros(c.nworkers)
    res = ag.eval_parallel(n, entropy=entropy)
    assert len(res) == n
    for r in res:
        assert r.reward == 20.0
        assert r.length == 3
    ag.close()


@pytest.mark.parametrize('make_ag', [A2cAgent, PpoAgent, A2ocAgent])
def test_nstep(make_ag: callable) -> None:
    c = rainy.Config()
    c.nworkers = 6
    c.nsteps = 4
    c.ppo_minibatch_size = 12
    c.set_parallel_env(rainy.envs.MultiProcEnv)
    c.set_net_fn('actor-critic', net.actor_critic.fc_shared(units=[32, 32]))
    c.set_net_fn('option-critic', net.option_critic.fc_shared(units=[32, 32]))
    c.set_env(partial(DummyEnvDeterministic, flatten=True))
    ag = make_ag(c)
    states = ag.penv.reset()
    ag._reset(states)
    states = ag.nstep(states)
    ag.close()
