import os

from torch import optim

import rainy
from rainy import Config
from rainy.agents import BootDQNAgent
from rainy.envs import DeepSea
from rainy.lib import explore
from rainy.net import bootstrap
from rainy.replay import BootDQNReplayFeed, UniformReplayBuffer


@rainy.main(BootDQNAgent, os.path.realpath(__file__))
def main(
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
    main()
