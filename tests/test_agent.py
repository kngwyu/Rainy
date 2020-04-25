from functools import partial

import pytest

import rainy
from rainy import net
from rainy.agents import A2CAgent, AOCAgent, PPOAgent
from test_env import DummyEnvDeterministic


@pytest.mark.parametrize(
    "n, make_ag", [(6, A2CAgent), (12, A2CAgent), (6, PPOAgent), (6, AOCAgent)]
)
def test_eval_parallel(n: int, make_ag: callable) -> None:
    c = rainy.Config()
    c.nworkers = 6
    c.nsteps = 5
    c.set_parallel_env(rainy.envs.DummyParallelEnv)
    c.set_net_fn("actor-critic", net.actor_critic.fc_shared(units=[32, 32]))
    c.set_net_fn("option-critic", net.option_critic.fc_shared(units=[32, 32]))
    c.set_env(partial(DummyEnvDeterministic, flatten=True))
    ag = make_ag(c)
    res = ag.eval_parallel(n=n)
    assert len(res) == n
    for r in res:
        assert r.return_ == 20.0
        assert r.length == 3
    ag.close()


@pytest.mark.parametrize("make_ag", [A2CAgent, PPOAgent, AOCAgent])
def test_nstep_train(make_ag: callable) -> None:
    c = rainy.Config()
    c.logger.setup_logdir()
    c.nworkers = 6
    c.nsteps = 4
    c.ppo_minibatch_size = 12
    c.set_parallel_env(rainy.envs.DummyParallelEnv)
    c.set_net_fn("actor-critic", net.actor_critic.fc_shared(units=[32, 32]))
    c.set_net_fn("option-critic", net.option_critic.fc_shared(units=[32, 32]))
    c.set_env(partial(DummyEnvDeterministic, flatten=True))
    ag = make_ag(c)
    res = next(ag.train_episodes(1))
    assert len(res) == c.nworkers
    ag.close()
