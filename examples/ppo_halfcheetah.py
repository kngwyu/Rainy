import os

from torch.optim import Adam

import rainy
from rainy.agents import PPOAgent
from rainy.envs import PyBullet, pybullet_parallel
from rainy.net.policy import SeparateStdGaussianDist


@rainy.main(PPOAgent, script_path=os.path.realpath(__file__))
def main(envname: str = "HalfCheetah") -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: PyBullet(envname))
    c.set_net_fn(
        "actor-critic", rainy.net.actor_critic.fc_shared(policy=SeparateStdGaussianDist)
    )
    c.set_parallel_env(pybullet_parallel())
    c.set_optimizer(lambda params: Adam(params, lr=3.0e-4, eps=1.0e-4))
    c.max_steps = int(2e6)
    c.grad_clip = 0.5
    # ppo params
    c.value_loss_weight = 0.5
    c.entropy_weight = 0.0
    c.gae_lambda = 0.95
    c.nworkers = 16
    c.nsteps = 128
    c.ppo_minibatch_size = (16 * 128) // 16
    c.ppo_clip = 0.2
    c.use_gae = True
    c.eval_freq = None
    return c


if __name__ == "__main__":
    main()
