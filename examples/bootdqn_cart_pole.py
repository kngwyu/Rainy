import os
from rainy import Config
from rainy.agents import BootDQNAgent
import rainy.utils.cli as cli


def config() -> Config:
    c = Config()
    c.max_steps = 100000
    c.episode_log_freq = 100
    return c


if __name__ == "__main__":
    cli.run_cli(config, BootDQNAgent, script_path=os.path.realpath(__file__))
