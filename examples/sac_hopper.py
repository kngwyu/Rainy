import click
import os
from rainy import Config
from rainy.agents import SACAgent
from rainy.envs import PyBullet
import rainy.utils.cli as cli
from torch.optim import Adam


def config(envname: str = "Hopper", nworkers: int = 1) -> Config:
    c = Config()
    c.set_env(lambda: PyBullet(envname))
    c.max_steps = int(1e6)
    c.set_optimizer(lambda params: Adam(params, lr=3e-4), key="actor")
    c.set_optimizer(lambda params: Adam(params, lr=3e-4), key="critic")
    c.set_optimizer(lambda params: Adam(params, lr=3e-4), key="entropy")
    c.replay_size = int(1e6)
    c.train_start = int(1e4)
    c.eval_deterministic = True
    c.eval_freq = c.max_steps // 100
    c.sync_freq = 1
    c.grad_clip = None
    c.nworkers = nworkers
    c.replay_batch_size = 256 * nworkers
    return c


if __name__ == "__main__":
    options = [click.Option(["--nworkers"], type=int, default=1)]
    cli.run_cli(config, SACAgent, os.path.realpath(__file__), options)
