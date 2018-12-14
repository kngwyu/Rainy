"""Train A2C agent in ALE game registerd in gym.
Some hyper parametes are from https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py
"""
import os
from rainy import Config, net
from rainy.agent import A2cAgent
from rainy.envs import Atari
import rainy.util.cli as cli
from rainy.envs import FrameStackParallel, MultiProcEnv
from torch.optim import RMSprop


def config() -> Config:
    c = Config()
    c.set_env(lambda: Atari('Breakout', frame_stack=False, frame_stack_parallel=True))
    c.set_optimizer(
        lambda params: RMSprop(params, lr=7e-4, alpha=0.99, eps=1e-5)
    )
    c.set_net_fn('actor-critic', net.actor_critic.ac_conv)
    c.num_workers = 16
    c.set_parallel_env(lambda env_gen, num_w: FrameStackParallel(MultiProcEnv(env_gen, num_w)))
    c.grad_clip = 0.5
    c.gae_tau = 1.0
    c.value_loss_weight = 0.5
    c.use_gae = False
    c.max_steps = int(8e7)
    c.eval_env = Atari('Breakout', episodic_life=False)
    c.eval_freq = None
    c.episode_log_freq = 1000
    return c


if __name__ == '__main__':
    cli.run_cli(config(), lambda c: A2cAgent(c), script_path=os.path.realpath(__file__))
