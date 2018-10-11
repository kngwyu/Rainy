import os
from rainy import Config
from rainy.agent import DqnAgent
import rainy.util.cli as cli


def config() -> Config:
    c = Config()
    c.max_steps = 100000
    c.double_q = True
    return c


if __name__ == '__main__':
    cli.run_cli(config(), lambda c: DqnAgent(c), script_path=os.path.realpath(__file__))
