"""Train A2OC agent in ALE game registerd in gym.
"""
import os

import click
from torch.optim import RMSprop

import rainy.utils.cli as cli
from rainy import Config, net
from rainy.agents import AOCAgent
from rainy.envs import Atari, atari_parallel


def config(
    envname: str = "Breakout",
    num_options: int = 4,
    opt_delib_cost: float = 0.025,
    opt_beta_adv_merginal: float = 0.01,
    use_gae: bool = False,
) -> Config:
    c = Config()
    c.set_env(lambda: Atari(envname, frame_stack=False))
    c.set_optimizer(lambda params: RMSprop(params, lr=7e-4, alpha=0.99, eps=1e-5))
    c.set_net_fn("option-critic", net.option_critic.conv_shared(num_options=4))
    c.nworkers = 16
    c.nsteps = 5
    c.set_parallel_env(atari_parallel())
    c.grad_clip = 0.5
    c.value_loss_weight = 1.0
    c.use_gae = use_gae
    c.max_steps = int(2e7)
    c.eval_env = Atari(envname)
    c.eval_deterministic = False
    c.episode_log_freq = 100
    c.opt_delib_cost = opt_delib_cost
    c.opt_beta_adv_merginal = opt_beta_adv_merginal
    c.eval_freq = c.max_steps // 10
    c.save_freq = None
    return c


if __name__ == "__main__":
    options = [
        click.Option(["--num-options"], type=int, default=4),
        click.Option(["--opt-delib-cost"], type=float, default=0.025),
        click.Option(["--opt-beta-adv-merginal"], type=float, default=0.01),
        click.Option(["--use-gae"], is_flag=True),
    ]
    cli.run_cli(config, AOCAgent, os.path.realpath(__file__), options=options)
