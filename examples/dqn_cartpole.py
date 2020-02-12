import os

import click

import rainy.utils.cli as cli
from rainy import Config
from rainy.agents import DQNAgent
from rainy.envs import ClassicControl, MultiProcEnv


def config(
    envname: str = "CartPole-v0", max_steps: int = 100000, nworkers: int = 1
) -> Config:
    c = Config()
    c.set_env(lambda: ClassicControl(envname))
    c.set_parallel_env(MultiProcEnv)
    c.max_steps = max_steps
    c.episode_log_freq = 100
    c.nworkers = nworkers
    c.replay_batch_size = 64 * nworkers
    return c


if __name__ == "__main__":
    options = [click.Option(["--nworkers"], type=int, default=1)]
    cli.run_cli(config, DQNAgent, os.path.realpath(__file__), options)
