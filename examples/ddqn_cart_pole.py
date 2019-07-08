import os
from rainy import Config
from rainy.agents import DoubleDqnAgent
import rainy.utils.cli as cli


def config() -> Config:
    c = Config()
    c.max_steps = 100000
    return c


if __name__ == '__main__':
    cli.run_cli(config, DoubleDqnAgent, script_path=os.path.realpath(__file__))
