from rainy import Config
from rainy.agent import DqnAgent
from rainy.util import run_cli


def config() -> Config:
    c = Config()
    c.max_steps = 100000
    c.double_q = True


if __name__ == '__main__':
    run_cli('dqn-cartpole', cconfig(), lambda c: DqnAgent(c))
