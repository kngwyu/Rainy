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
    opt_beta_adv_terminal: float = 0.01,
    use_gae: bool = False,
) -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: ClassicControl(envname))
    c.max_steps = int(4e5)
    c.nworkers = 12
    c.nsteps = 5
    c.set_parallel_env(MultiProcEnv)
    c.set_optimizer(lambda params: optim.RMSprop(params, lr=0.0007))
    c.grad_clip = 1.0
    c.eval_freq = None
    c.entropy_weight = 0.001
    c.value_loss_weight = 1.0
    c.opt_delib_cost = 0.0
    c.opt_beta_adv_merginal = 0.01
    c.use_gae = use_gae
    c.set_net_fn(
        "option-critic", rainy.net.option_critic.fc_shared(num_options=num_options)
    )
    return c


if __name__ == "__main__":
    options = [
        click.Option(["--num-options"], type=int, default=2),
        click.Option(["--opt-delib-cost"], type=float, default=0.0),
        click.Option(["--opt-beta-adv-terminal"], type=float, default=0.01),
        click.Option(["--use-gae"], is_flag=True),
    ]
    run_cli(config, rainy.agents.AOCAgent, os.path.realpath(__file__), options)
