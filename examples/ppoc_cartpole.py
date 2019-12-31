import click
import os
import rainy
from rainy.utils.cli import run_cli
from rainy.envs import ClassicControl, MultiProcEnv
from torch import optim


def config(
    envname: str = "CartPole-v0",
    num_options: int = 2,
    opt_delib_cost: float = 0.0,
    opt_beta_adv_merginal: float = 0.01,
) -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: ClassicControl(envname))
    c.max_steps = int(4e5)
    # Option settings
    c.opt_delib_cost = opt_delib_cost
    c.opt_beta_adv_merginal = opt_beta_adv_merginal
    c.set_net_fn(
        "option-critic", rainy.net.option_critic.fc_shared(num_options=num_options)
    )
    # PPO params
    c.nworkers = 12
    c.nsteps = 64
    c.set_parallel_env(MultiProcEnv)
    c.set_optimizer(lambda params: optim.Adam(params, lr=2.5e-4, eps=1.0e-4))
    c.grad_clip = 1.0
    c.eval_freq = 10000
    c.entropy_weight = 0.01
    c.value_loss_weight = 1.0
    c.use_gae = True
    return c


if __name__ == "__main__":
    options = [
        click.Option(["--num-options"], type=int, default=2),
        click.Option(["--opt-delib-cost"], type=float, default=0.0),
        click.Option(["--opt-beta-adv-merginal"], type=float, default=0.01),
    ]
    run_cli(config, rainy.agents.PPOCAgent, os.path.realpath(__file__), options)
