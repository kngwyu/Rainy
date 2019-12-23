import os
from rainy import Config
from rainy.agents import DQNAgent
from rainy.envs import ClassicControl
import rainy.utils.cli as cli


def config(envname: str = "CartPole-v0", max_steps: int = 100000) -> Config:
    c = Config()
    c.set_env(lambda: ClassicControl(envname))
    c.max_steps = max_steps
    c.episode_log_freq = 100
    return c


if __name__ == "__main__":
    cli.run_cli(config, DQNAgent, script_path=os.path.realpath(__file__))
