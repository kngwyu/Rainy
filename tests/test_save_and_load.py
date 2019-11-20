from rainy import Config
from rainy.agents import PpoAgent
from rainy.envs import MultiProcEnv
from pathlib import Path
import pytest
from torch.optim import Adam

LOG_DIR = Path("/tmp/rainy-test/")


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
    c.logger.setup()
    return c


@pytest.mark.filterwarnings("ignore:PkgResourcesDeprecationWarning")
def test_ppo_save() -> None:
    ppo = PpoAgent(config())
    ppo.optimizer.param_groups[0]["lr"] = 1.0
    ppo.clip_eps = 0.2
    ppo.save("ppo-agent.pth")
    ppo.close()
    ppo = PpoAgent(config())
    path = ppo.config.logger.logdir.joinpath("ppo-agent.pth")
    ppo.load(path.as_posix())
    assert ppo.clip_eps == 0.2
    assert ppo.optimizer.param_groups[0]["lr"] == 1.0
    ppo.close()
