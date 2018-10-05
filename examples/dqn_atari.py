import os
from rainy import agent, Config, net, train_agent
from rainy.agent import Agent
from rainy.env_ext import Atari
from rainy.explore import EpsGreedy, LinearCooler
from torch.optim import RMSprop


def dqn_atari() -> Agent:
    c = Config()
    c.set_env(lambda: Atari('Breakout'))
    c.set_optimizer(
        lambda params: RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    )
    c.set_explorer(
        lambda net: EpsGreedy(1.0, LinearCooler(1.0, 0.1, int(1e6)), net)
    )
    c.double_q = True
    c.set_value_net(net.value_net.dqn_conv)
    c.replay_size = int(1e6)
    c.batch_size = 32
    c.train_start = 5000
    c.sync_freq = 10000
    c.max_steps = int(2e7)
    c.eval_env = Atari('Breakout', episode_life=False)
    c.eval_freq = None
    c.logger.set_dir_from_script_path(os.path.realpath(__file__))
    c.logger.set_stderr()
    a = agent.DqnAgent(c)
    return a


if __name__ == '__main__':
    ag = dqn_atari()
    train_agent(ag)
