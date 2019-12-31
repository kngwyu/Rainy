import click
import os
from rainy import Config
from rainy.agents import BootDQNAgent
from rainy.envs import DeepSea
from rainy.lib import explore
from rainy.net import bootstrap
from rainy.replay import BootDQNReplayFeed, UniformReplayBuffer
import rainy.utils.cli as cli
from torch import optim


def config(
    max_steps: int = 100000,
    size: int = 20,
    rpf: bool = False,
    replay_prob: float = 0.5,
    prior_scale: float = 1.0,
) -> Config:
    c = Config()
    c.set_optimizer(lambda params: optim.Adam(params))
    c.set_explorer(lambda: explore.Greedy())
    c.set_explorer(lambda: explore.Greedy(), key="eval")
    c.set_env(lambda: DeepSea(size))
    c.max_steps = max_steps
    c.episode_log_freq = 100
    c.replay_prob = replay_prob
    c.update_freq = size
    c.train_start = 100
    if rpf:
        c.set_net_fn("bootdqn", bootstrap.rpf_fc_separated(10, prior_scale=prior_scale))
    c.set_replay_buffer(
        lambda capacity: UniformReplayBuffer(
            BootDQNReplayFeed, capacity=capacity, allow_overlap=True
        )
    )
    return c


if __name__ == "__main__":
    options = [
        click.Option(["--rpf"], is_flag=True),
        click.Option(["--replay-prob", "-RP"], type=float, default=0.5),
        click.Option(["--prior-scale", "-PS"], type=float, default=1.0),
    ]
    cli.run_cli(config, BootDQNAgent, os.path.realpath(__file__), options)
