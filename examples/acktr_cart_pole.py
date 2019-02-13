import os
from rainy import Config
from rainy.agents import AcktrAgent
import rainy.utils.cli as cli
from rainy.envs import MultiProcEnv
from rainy.lib import kfac


def config() -> Config:
    c = Config()
    c.max_steps = int(1e6)
    c.nworkers = 12
    c.nsteps = 10
    c.set_parallel_env(lambda env_gen, num_w: MultiProcEnv(env_gen, num_w))
    c.set_optimizer(kfac.default_sgd(eta_max=0.2))
    c.set_preconditioner(lambda net: kfac.KfacPreConditioner(net, eta_max=0.1, tau=120.))
    c.gae_tau = 0.95
    c.use_gae = True
    c.eval_deterministic = True
    c.lr_decay = True
    c.value_loss_weight = 0.1
    c.entropy_weight = 0.01
    return c


if __name__ == '__main__':
    cli.run_cli(config(), AcktrAgent, script_path=os.path.realpath(__file__))
