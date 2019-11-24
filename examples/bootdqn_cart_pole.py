import click
import os
from rainy import Config
from rainy.agents import BootDQNAgent
from rainy.envs import ClassicControl
from rainy.lib import explore
from rainy.net import bootstrap
from rainy.replay import BootDQNReplayFeed, UniformReplayBuffer
import rainy.utils.cli as cli
from torch import optim


def config(
    envname: str = "CartPole-v0",
    max_steps: int = 1000000,
    rpf: bool = False,
    replay_prob: float = 0.5,
    prior_scale: float = 1.0,
) -> Config:
    c = Config()
    c.set_optimizer(lambda params: optim.Adam(params))
    c.set_explorer(lambda: explore.Greedy())
    c.set_explorer(lambda: explore.Greedy(), key="eval")
    c.set_env(lambda: ClassicControl(envname))
    c.max_steps = max_steps
    c.episode_log_freq = 100
    if rpf:
        c.set_net_fn("bootdqn", bootstrap.rpf_fc_separated(10, prior_scale=prior_scale))
    c.set_replay_buffer(
        lambda capacity: UniformReplayBuffer(BootDQNReplayFeed, capacity=capacity)
    )
    return c


if __name__ == "__main__":
    options = [
        click.Option(["--rpf"], is_flag=True),
        click.Option(["--replay-prob", "-RP"], type=float, default=0.5),
        click.Option(["--prior-scale", "-PS"], type=float, default=1.0),
    ]
    cli.run_cli(config, BootDQNAgent, os.path.realpath(__file__), options)
