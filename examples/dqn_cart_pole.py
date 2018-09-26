import os
from rainy import Config, train_agent
from rainy.agent import Agent, DqnAgent


def dqn_cartpole() -> Agent:
    c = Config()
    c.max_steps = 100000
    c.double_q = True
    a = DqnAgent(c)
    path = os.path.realpath(__file__)
    c.logger.set_dir_from_script_path(path)
    c.logger.set_stderr()
    c.log_freq = 100
    return a


if __name__ == '__main__':
    ag = dqn_cartpole()
    train_agent(ag)
