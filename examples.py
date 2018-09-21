import numpy as np
from rainy import agent, Config, net
from rainy.agent import Agent
from rainy.env_ext import Atari, EnvExt
from rainy.explore import EpsGreedy, LinearCooler
from typing import Optional
from torch.optim import RMSprop

def run_agent(ag: Agent, eval_env: Optional[EnvExt] = None):
    max_steps = ag.config.max_steps
    turn = 0
    rewards_sum = 0
    while True:
        if max_steps and ag.total_steps > max_steps:
            break
        if turn % 100 == 0:
            print('turn: {}, total_steps: {}, rewards: {}'.format(
                turn,
                ag.total_steps,
                rewards_sum
            ))
            rewards_sum = 0
        if turn % 1000 == 0:
            print('eval: {}'.format(ag.eval_episode(eval_env=eval_env)))
        rewards_sum += ag.episode()
        turn += 1
        print(turn)
    ag.save("saved-example.rainy")


def run():
    c = Config()
    c.max_steps = 100000
    a = agent.DqnAgent(c)
    # a.load("saved-example")
    run_agent(a)


def run_atari():
    c = Config()
    c.set_env(lambda: Atari('Breakout', frame_stack=True))
    c.set_optimizer(
        lambda params: RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    )
    c.set_explorer(
        lambda net: EpsGreedy(1.0, LinearCooler(1.0, 0.1, int(1e6)), net)
    )
    c.set_value_net(net.value_net.dqn_conv)
    c.replay_size = int(1e6)
    c.batch_size = 32
    c.train_start = 50000
    c.sync_freq = 10000
    c.max_steps = int(2e7)
    a = agent.DqnAgent(c)
    # a.load("saved-example")
    eval_env = Atari('Breakout', frame_stack=True, episode_life=False)
    run_agent(a, eval_env=eval_env)


if __name__ == '__main__':
    run_atari()
