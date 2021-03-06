import os

from rainy import Config
from rainy.agents import DoubleDQNAgent
from rainy.envs import ClassicControl


@rainy.main(DoubleDQNAgent, script_path=os.path.realpath(__file__))
def main(envname: str = "CartPole-v0") -> Config:
    c = Config()
    c.set_env(lambda: ClassicControl(envname))
    c.max_steps = 100000
    return c


if __name__ == "__main__":
    cli.run_cli(config, DoubleDQNAgent, script_path=os.path.realpath(__file__))
