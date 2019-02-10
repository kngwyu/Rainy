import os
from rainy import Config
from rainy.agents import PpoAgent
import rainy.utils.cli as cli
from rainy.envs import MultiProcEnv
from torch.optim import Adam


def config() -> Config:
    c = Config()
    c.max_steps = int(1e6)
    c.nworkers = 8
    c.set_parallel_env(lambda env_gen, num_w: MultiProcEnv(env_gen, num_w))
    c.set_optimizer(lambda params: Adam(params, lr=2.5e-4, eps=1.0e-4))
    c.value_loss_weight = 0.2
    c.network_log_freq = 20
    c.grad_clip = 0.1
    c.gae_tau = 0.95
    c.nsteps = 128
    c.ppo_minibatch_size = 32
    c.use_gae = True
    c.lr_decay = True
    c.clip_decay = True
    return c


if __name__ == '__main__':
    cli.run_cli(config(), PpoAgent, script_path=os.path.realpath(__file__))
