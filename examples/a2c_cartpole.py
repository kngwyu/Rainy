import click
import os
import rainy
from rainy.utils.cli import run_cli
from rainy.envs import ClassicControl, MultiProcEnv
from torch.optim import Adam


def config(
    envname: str = "CartPole-v0", rnn: bool = False, use_gae: bool = False
) -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: ClassicControl(envname))
    c.max_steps = int(1e6)
    c.nworkers = 12
    c.nsteps = 5
    c.set_parallel_env(MultiProcEnv)
    c.set_optimizer(lambda params: Adam(params, lr=0.001))
    c.grad_clip = 0.1
    c.value_loss_weight = 0.2
    c.use_gae = use_gae
    c.eval_deterministic = True
    c.eval_freq = c.max_steps // 10
    c.entropy_weight = 0.001
    if rnn:
        c.set_net_fn(
            "actor-critic", rainy.net.actor_critic.fc_shared(rnn=rainy.net.GruBlock)
        )
    return c


if __name__ == "__main__":
    options = [
        click.Option(["--rnn"], is_flag=True),
        click.Option(["--use-gae"], is_flag=True),
    ]
    run_cli(config, rainy.agents.A2CAgent, os.path.realpath(__file__), options)
