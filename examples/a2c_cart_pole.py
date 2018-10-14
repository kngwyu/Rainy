import os
from rainy import Config
from rainy.agent import A2cAgent
import rainy.util.cli as cli
from rainy.envs import DummyParallelEnv


def config() -> Config:
    c = Config()
    c.max_steps = 100000
    c.num_workers = 5
    c.set_parallel_env(lambda env_gen, num_w: DummyParallelEnv(env_gen, num_w))
    return c


if __name__ == '__main__':
    cli.run_cli(config(), lambda c: A2cAgent(c), script_path=os.path.realpath(__file__))
