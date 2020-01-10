import click
import os
import rainy
from rainy.net import termination_critic as tc
from rainy.utils.cli import run_cli
from rainy.envs import ClassicControl, MultiProcEnv
from torch import optim


def config(envname: str = "CartPole-v0", num_options: int = 2) -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: ClassicControl(envname))
    c.max_steps = int(4e5)
    c.nworkers = 12
    c.nsteps = 5
    c.set_parallel_env(MultiProcEnv)
    c.set_optimizer(lambda params: optim.Adam(params))
    c.set_optimizer(lambda params: optim.Adam(params), key="termination")
    c.set_explorer(lambda: rainy.lib.explore.EpsGreedy(0.1))
    c.grad_clip = 0.5
    c.eval_freq = 10000
    c.network_log_freq = (c.max_steps // c.batch_size) // 10
    c.entropy_weight = 0.001
    c.value_loss_weight = 1.0
    c.set_net_fn(
        "actor-critic", tc.oac_fc_shared(num_options=num_options),
    )
    c.set_net_fn(
        "termination-critic", tc.tc_fc_shared(num_options=num_options),
    )
    return c


if __name__ == "__main__":
    options = [
        click.Option(["--num-options"], type=int, default=2),
    ]
    run_cli(config, rainy.agents.ACTCAgent, os.path.realpath(__file__), options)
