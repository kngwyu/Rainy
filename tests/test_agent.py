from functools import partial
import numpy as np
import pytest
import rainy
from test_env import DummyEnvDeterministic


@pytest.mark.parametrize('n', [6, 12])
def test_eval_parallel(n: int) -> None:
    c = rainy.Config()
    c.nworkers = 6
    c.nsteps = 5
    c.set_parallel_env(rainy.envs.MultiProcEnv)
    c.set_net_fn('actor-critic', rainy.net.actor_critic.fc_shared(units=[32, 32]))
    c.set_env(partial(DummyEnvDeterministic, flatten=True))
    agent = rainy.agents.A2cAgent(c)
    entropy = np.zeros(c.nworkers)
    res = agent.eval_parallel(n, entropy=entropy)
    assert len(res) == n
    for r in res:
        assert r.reward == 20.0
        assert r.length == 3
    agent.close()

