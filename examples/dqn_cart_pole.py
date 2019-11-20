import os
from rainy import Config
from rainy.agents import DQNAgent
import rainy.utils.cli as cli


def config() -> Config:
    c = Config()
    c.max_steps = 100000
    c.eval_freq = 100
    return c


if __name__ == "__main__":
    cli.run_cli(config, DQNAgent, script_path=os.path.realpath(__file__))
