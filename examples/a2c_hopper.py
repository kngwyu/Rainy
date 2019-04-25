import os
from rainy import Config, net
from rainy.agents import A2cAgent
import rainy.utils.cli as cli
from rainy.envs import PyBullet, pybullet_parallel
from rainy.net.policy import SeparateStdGaussinanHead
from torch.optim import Adam


def config() -> Config:
    c = Config()
    c.set_env(lambda: PyBullet('Hopper'))
    c.set_net_fn('actor-critic', net.actor_critic.fc_shared(policy=SeparateStdGaussinanHead))
    c.set_parallel_env(pybullet_parallel())
    c.max_steps = int(1e6)
    c.nworkers = 12
    c.nsteps = 5
    c.set_optimizer(lambda params: Adam(params, lr=0.001))
    c.grad_clip = 0.5
    c.gae_lambda = 0.95
    c.value_loss_weight = 0.5
    c.entropy_weight = 0.0
    c.use_gae = False
    c.eval_deterministic = False
    c.eval_freq = None
    return c


if __name__ == '__main__':
    cli.run_cli(config(), A2cAgent, script_path=os.path.realpath(__file__))
