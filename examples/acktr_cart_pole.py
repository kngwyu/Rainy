import os
from rainy import Config
from rainy.agents import ACKTRAgent
import rainy.utils.cli as cli
from rainy.envs import MultiProcEnv
from rainy.lib import kfac


KFAC_KWARGS = {
    "tau": 12 * 20,
    "update_freq": 10,
    "norm_scaler": kfac.SquaredFisherScaler(eta_max=0.1, delta=0.001),
}


def config() -> Config:
    c = Config()
    c.max_steps = int(4e5)
    c.nworkers = 12
    c.nsteps = 20
    c.set_parallel_env(MultiProcEnv)
    c.set_optimizer(kfac.default_sgd(eta_max=0.1))
    c.set_preconditioner(lambda net: kfac.KfacPreConditioner(net, **KFAC_KWARGS))
    c.gae_lambda = 0.95
    c.use_gae = False
    c.lr_min = 0.0
    c.value_loss_weight = 0.2
    c.entropy_weight = 0.01
    c.eval_freq = None
    return c


if __name__ == "__main__":
    cli.run_cli(config, ACKTRAgent, script_path=os.path.realpath(__file__))
