import os
from rainy import Config
from rainy.agents import A2cAgent
import rainy.utils.cli as cli
from rainy.envs import MultiProcEnv
from torch.optim import Adam


def config() -> Config:
    c = Config()
    c.max_steps = int(1e6)
    c.nworkers = 12
    c.set_parallel_env(lambda env_gen, num_w: MultiProcEnv(env_gen, num_w))
    c.set_optimizer(lambda params: Adam(params, lr=0.001))
    c.grad_clip = 0.5
    c.gae_tau = 0.95
    c.use_gae = False
    c.eval_deterministic = True
    c.value_loss_weight = 0.1
    c.entropy_weight = 0.001
    return c


if __name__ == '__main__':
    cli.run_cli(config(), A2cAgent, script_path=os.path.realpath(__file__))
