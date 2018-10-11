import os
from rainy import agent, Config, net, train_agent
from rainy.agent import Agent
from rainy.env_ext import Atari
from rainy.explore import EpsGreedy, LinearCooler
from torch.optim import RMSprop


def config() -> Config:
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
    c.train_start = 50000
    c.sync_freq = 10000
    c.max_steps = int(2e7)
    c.eval_env = Atari('Breakout', episode_life=False)
    c.eval_freq = None
    return c


if __name__ == '__main__':
    cli.run_cli(config(), lambda c: DqnAgent(c), script_path=os.path.realpath(__file__))

