import os
from rainy import Config
from rainy.agents import AcktrAgent
import rainy.utils.cli as cli
from rainy.envs import MultiProcEnv
from rainy.lib import kfac


KFAC_KWARGS = {
    'delta': 0.005,
    'eta_max': 0.2,
    'tau': 120,
    'update_freq': 4,
}


def config() -> Config:
    c = Config()
    c.max_steps = int(4e5)
    c.nworkers = 12
    c.nsteps = 10
    c.set_parallel_env(lambda env_gen, num_w: MultiProcEnv(env_gen, num_w))
    c.set_optimizer(kfac.default_sgd(KFAC_KWARGS['eta_max']))
    c.set_preconditioner(lambda net: kfac.KFAC(net, eps=0.01, pi=True, constraint_norm=True))
    c.gae_tau = 0.95
    c.use_gae = True
    c.lr_decay = False
    c.eval_deterministic = True
    c.value_loss_weight = 0.1
    c.entropy_weight = 0.01
    return c


if __name__ == '__main__':
    cli.run_cli(config(), AcktrAgent, script_path=os.path.realpath(__file__))
