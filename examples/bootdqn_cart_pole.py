import click
import os
from rainy import Config
from rainy.agents import BootDQNAgent
from rainy.envs import ClassicControl
from rainy.net import bootstrap
from rainy.replay import BootDQNReplayFeed, UniformReplayBuffer
import rainy.utils.cli as cli


def config(envname: str = "CartPole-v0", rpf: bool = False) -> Config:
    c = Config()
    c.set_env(lambda: ClassicControl(envname))
    c.max_steps = 100000
    c.episode_log_freq = 100
    if rpf:
        c.set_net_fn("bootdqn", bootstrap.rpf_fc_separated(10))
    c.set_replay_buffer(
        lambda capacity: UniformReplayBuffer(BootDQNReplayFeed, capacity=capacity)
    )
    return c


if __name__ == "__main__":
    options = [click.Option(["--rpf"], is_flag=True)]
    cli.run_cli(config, BootDQNAgent, os.path.realpath(__file__), options)
