import os
from rainy import Config, envs
from rainy.net import bootstrap
from rainy.agents import BootDQNAgent
import rainy.utils.cli as cli


def config() -> Config:
    c = Config()
    c.max_steps = 100000
    c.episode_log_freq = 100
    c.set_net_fn("bootdqn", bootstrap.rpf_fc_separated(10))
    c.set_env(lambda: envs.ClassicalControl("MountainCar-v0"))
    return c


if __name__ == "__main__":
    cli.run_cli(config, BootDQNAgent, script_path=os.path.realpath(__file__))
