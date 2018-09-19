import numpy as np
from rainy import agent, Config, net
from rainy.agent import Agent
from rainy.env_ext import Atari
from rainy.explore import EpsGreedy, LinearCooler
from torch.optim import RMSprop

def run_agent(ag: Agent, train: bool = True):
    max_steps = ag.config.max_steps
    turn = 0
    while True:
        if max_steps and ag.total_steps > max_steps:
            break
        if turn % 100 == 0:
            print(ag.episode(train=train))
        else:
            ag.episode(train=train)
        turn += 1
    ag.save("saved-example.rainy")


def run():
    c = Config()
    c.max_steps = 100000
    a = agent.DqnAgent(c)
    # a.load("saved-example")
    run_agent(a)

def run_atari():
    c = Config()
    c.set_env(lambda: Atari('Breakout'))
    c.set_optimizer(
        lambda params: RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    )
    c.set_explorer(
        lambda net: EpsGreedy(1.0, LinearCooler(1.0, 0.1, 1e6), net)
    )
    c.set_value_net(net.value_net.dqn_conv)
    c.max_steps = 10000
    c.replay_size = 1e6
    c.batch_size = 32
    c.set_wrap_states(np.vectorize(lambda x: x / 255.0))
    c.train_start = 50000
    c.max_steps = 1e7
    a = agent.DqnAgent(c)
    # a.load("saved-example")
    run_agent(a)


if __name__ == '__main__':
    run_atari()
