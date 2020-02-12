import os

import click
from torch import optim

import rainy
from rainy.net import termination_critic as tc
from rainy.utils.cli import run_cli


def config(envname: str = "Breakout", num_options: int = 4) -> rainy.Config:
    c = rainy.Config()

    c.set_env(lambda: rainy.envs.Atari(envname, frame_stack=False))
    c.set_parallel_env(rainy.envs.atari_parallel())
    c.eval_env = rainy.envs.Atari(envname)

    c.max_steps = int(2e7)
    c.nworkers = 16
    c.nsteps = 5

    c.grad_clip = 1.0
    c.eval_freq = c.max_steps // 20
    c.network_log_freq = (c.max_steps // c.batch_size) // 10
    c.entropy_weight = 0.001
    c.value_loss_weight = 1.0

    c.set_optimizer(lambda params: optim.RMSprop(params, lr=7e-4, alpha=0.99, eps=1e-5))
    c.set_optimizer(lambda params: optim.Adam(params, lr=1e-4), key="termination")
    c.set_net_fn("actor-critic", tc.oac_conv_shared(num_options=num_options))
    c.set_net_fn("termination-critic", tc.tc_conv_shared(num_options=num_options))
    c.save_freq = None
    return c


if __name__ == "__main__":
    options = [
        click.Option(["--num-options"], type=int, default=4),
    ]
    run_cli(config, rainy.agents.ACTCAgent, os.path.realpath(__file__), options)
