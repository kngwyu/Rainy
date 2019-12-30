import click
import os
from rainy import Config
from rainy.agents import DDPGAgent
from rainy.envs import PyBullet
from rainy.lib import explore
import rainy.utils.cli as cli
from torch.optim import Adam


def config(envname: str = "Hopper", nworkers: int = 1) -> Config:
    c = Config()
    c.set_env(lambda: PyBullet(envname))
    c.max_steps = int(1e6)
    c.set_optimizer(lambda params: Adam(params, lr=1e-3), key="actor")
    c.set_optimizer(lambda params: Adam(params, lr=1e-3), key="critic")
    c.replay_size = int(1e6)
    c.train_start = int(1e4)
    c.set_explorer(lambda: explore.GaussianNoise())
    c.set_explorer(lambda: explore.Greedy(), key="eval")
    c.use_reward_monitor = True
    c.eval_deterministic = True
    c.grad_clip = None
    c.eval_freq = c.max_steps // 10
    c.nworkers = nworkers
    c.replay_batch_size = 100 * nworkers
    return c


if __name__ == "__main__":
    options = [click.Option(["--nworkers"], type=int, default=1)]
    cli.run_cli(config, DDPGAgent, os.path.realpath(__file__), options)
