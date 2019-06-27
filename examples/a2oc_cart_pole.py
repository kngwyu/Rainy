import os
import rainy
from rainy.utils.cli import run_cli
from rainy.envs import MultiProcEnv
from torch.optim import Adam


def config() -> rainy.Config:
    c = rainy.Config()
    c.max_steps = int(1e5)
    c.nworkers = 6
    c.nsteps = 5
    c.set_parallel_env(MultiProcEnv)
    c.set_optimizer(lambda params: Adam(params, lr=0.001))
    c.grad_clip = 0.5
    c.eval_freq = None
    c.entropy_weight = 0.001
    c.opt_epsilon_init = 1.0
    c.opt_epsilon_minimal = 0.1
    return c


if __name__ == '__main__':
    run_cli(config(), rainy.agents.A2ocAgent, script_path=os.path.realpath(__file__))
