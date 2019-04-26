"""Train A2C agent in ALE game registerd in gym.
Some hyper parametes are from https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py
"""
import os
from rainy import Config, net
from rainy.agents import A2cAgent
from rainy.envs import Atari, atari_parallel
import rainy.utils.cli as cli
from torch.optim import RMSprop


def config() -> Config:
    c = Config()
    c.set_env(lambda: Atari('Breakout', frame_stack=False))
    c.set_optimizer(
        lambda params: RMSprop(params, lr=7e-4, alpha=0.99, eps=1e-5)
    )
    #  c.set_net_fn('actor-critic', net.actor_critic.ac_conv(rnn=net.GruBlock))
    c.set_net_fn('actor-critic', net.actor_critic.ac_conv())
    c.nworkers = 16
    c.nsteps = 5
    c.set_parallel_env(atari_parallel())
    c.grad_clip = 0.5
    c.value_loss_weight = 0.5
    c.use_gae = False
    c.max_steps = int(2e7)
    c.eval_env = Atari('Breakout')
    c.use_reward_monitor = True
    c.eval_deterministic = False
    c.episode_log_freq = 100
    c.eval_freq = None
    c.save_freq = None
    return c


if __name__ == '__main__':
    cli.run_cli(config(), A2cAgent, script_path=os.path.realpath(__file__))
