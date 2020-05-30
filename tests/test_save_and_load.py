from pathlib import Path

import pytest
from torch.optim import Adam

from rainy import Config
from rainy.agents import PPOAgent
from rainy.envs import MultiProcEnv, PyBullet, pybullet_parallel

LOG_DIR = Path("/tmp/rainy-test/")

try:
    import pybullet  # noqa

    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False


def config() -> Config:
    c = Config()
    c.max_steps = int(1e6)
    c.nworkers = 8
    c.set_parallel_env(lambda env_gen, num_w: MultiProcEnv(env_gen, num_w))
    c.set_optimizer(lambda params: Adam(params, lr=2.5e-4, eps=1.0e-4))
    c.ppo_clip = 0.5
    if not LOG_DIR.exists():
        LOG_DIR.mkdir()
    c.logger.logdir = Path(LOG_DIR)
    c.logger.setup_logdir()
    return c


@pytest.mark.filterwarnings("ignore:PkgResourcesDeprecationWarning")
def test_ppo_save() -> None:
    ppo = PPOAgent(config())
    ppo.optimizer.param_groups[0]["lr"] = 1.0
    ppo.clip_eps = 0.2
    ppo.save("ppo-agent.pth")
    ppo.close()
    ppo = PPOAgent(config())
    path = ppo.config.logger.logdir.joinpath("ppo-agent.pth")
    ppo.load(path.as_posix())
    assert ppo.clip_eps == 0.2
    assert ppo.optimizer.param_groups[0]["lr"] == 1.0
    ppo.close()


@pytest.mark.filterwarnings("ignore:PkgResourcesDeprecationWarning")
@pytest.mark.skipif(not HAS_PYBULLET, reason="PyBullet is an optional dependency")
def test_rms_save() -> None:
    c = config()
    c.set_env(lambda: PyBullet())
    c.set_parallel_env(pybullet_parallel())
    ppo = PPOAgent(c)
    ppo.penv.as_cls("NormalizeObsParallel")._rms.mean = 10.0
    ppo.save("ppo-agent.pth")
    ppo.close()
    ppo = PPOAgent(c)
    path = ppo.config.logger.logdir.joinpath("ppo-agent.pth")
    ppo.load(path.as_posix())
    mean = ppo.penv.as_cls("NormalizeObsParallel")._rms.mean.mean()
    assert 9.999 <= mean <= 10.001
    ppo.close()
