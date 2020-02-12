import os

import rainy.utils.cli as cli
from rainy import Config, net
from rainy.agents import ACKTRAgent
from rainy.envs import PyBullet, pybullet_parallel
from rainy.lib import kfac
from rainy.net.policy import SeparateStdGaussianDist

KFAC_KWARGS = {
    "tau": 12 * 20,
    "update_freq": 10,
    "norm_scaler": kfac.SquaredFisherScaler(eta_max=0.1, delta=0.001),
}


def config(envname: str = "Hopper") -> Config:
    c = Config()
    c.max_steps = int(4e5)
    c.nworkers = 12
    c.nsteps = 20
    c.set_env(lambda: PyBullet(envname))
    c.set_net_fn(
        "actor-critic", net.actor_critic.fc_shared(policy=SeparateStdGaussianDist)
    )
    c.set_parallel_env(pybullet_parallel())
    c.set_optimizer(kfac.default_sgd(eta_max=0.1))
    c.set_preconditioner(lambda net: kfac.KfacPreConditioner(net, **KFAC_KWARGS))
    c.gae_lambda = 0.95
    c.use_gae = True
    c.eval_deterministic = False
    c.value_loss_weight = 0.5
    c.entropy_weight = 0.0
    c.eval_freq = None
    return c


if __name__ == "__main__":
    cli.run_cli(config, ACKTRAgent, script_path=os.path.realpath(__file__))
