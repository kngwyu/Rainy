import os
from rainy import Config, net
from rainy.agents import PpoAgent
import rainy.utils.cli as cli
from rainy.envs import PyBullet, pybullet_parallel
from rainy.net.policy import SeparateStdGaussianDist
from torch.optim import Adam


def config() -> Config:
    c = Config()
    c.set_env(lambda: PyBullet('Hopper'))
    c.set_net_fn('actor-critic', net.actor_critic.fc_shared(policy=SeparateStdGaussianDist))
    c.set_parallel_env(pybullet_parallel())
    c.set_optimizer(lambda params: Adam(params, lr=3.0e-4, eps=1.0e-4))
    c.max_steps = int(2e6)
    c.grad_clip = 0.5
    # ppo params
    c.value_loss_weight = 0.5
    c.entropy_weight = 0.0
    c.gae_lambda = 0.95
    c.nworkers = 4
    c.nsteps = 512
    c.ppo_minibatch_size = (4 * 512) // 8
    c.ppo_clip = 0.2
    c.use_gae = True
    c.use_reward_monitor = True
    c.eval_freq = None
    return c


if __name__ == '__main__':
    cli.run_cli(config(), PpoAgent, script_path=os.path.realpath(__file__))
