import os
from rainy import Config, net
from rainy.agents import DqnAgent
from rainy.envs import Atari
from rainy.lib.explore import EpsGreedy, LinearCooler
import rainy.utils.cli as cli
from torch.optim import RMSprop


def config() -> Config:
    c = Config()
    c.set_env(lambda: Atari('Breakout'))
    c.set_optimizer(
        lambda params: RMSprop(params, lr=0.00025, alpha=0.95, eps=0.01, centered=True)
    )
    c.set_explorer(lambda: EpsGreedy(1.0, LinearCooler(1.0, 0.1, int(1e6))))
    c.set_net_fn('value', net.value.dqn_conv())
    c.replay_size = int(1e6)
    c.batch_size = 32
    c.train_start = 50000
    c.sync_freq = 10000
    c.max_steps = int(2e7)
    c.eval_env = Atari('Breakout', episodic_life=False)
    c.eval_freq = None
    c.seed = 1
    c.use_reward_monitor = True
    return c


if __name__ == '__main__':
    cli.run_cli(config(), DqnAgent, script_path=os.path.realpath(__file__))

