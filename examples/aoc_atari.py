"""Train A2OC agent in ALE game registerd in gym.
"""
import os
from rainy import Config, net
from rainy.agents import AocAgent
from rainy.envs import Atari, atari_parallel
from rainy.lib.explore import DummyCooler, EpsGreedy
import rainy.utils.cli as cli
from torch.optim import RMSprop


def config(game: str = "Breakout") -> Config:
    c = Config()
    c.set_env(lambda: Atari(game, frame_stack=False))
    c.set_optimizer(lambda params: RMSprop(params, lr=7e-4, alpha=0.99, eps=1e-5))
    c.set_explorer(lambda: EpsGreedy(1.0, DummyCooler(0.1)))
    c.set_net_fn("option-critic", net.option_critic.conv_shared(num_options=4))
    c.nworkers = 16
    c.nsteps = 5
    c.set_parallel_env(atari_parallel())
    c.grad_clip = 0.5
    c.value_loss_weight = 1.0
    c.use_gae = False
    c.max_steps = int(2e7)
    c.eval_env = Atari(game)
    c.use_reward_monitor = True
    c.eval_deterministic = False
    c.episode_log_freq = 100
    c.opt_delib_cost = 0.025
    c.opt_beta_adv_merginal = 0.01
    c.eval_freq = None
    c.save_freq = None
    return c


if __name__ == "__main__":
    cli.run_cli(config, AocAgent, script_path=os.path.realpath(__file__))
