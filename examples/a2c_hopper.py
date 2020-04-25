import os

from torch.optim import Adam

import rainy
from rainy.agents import A2CAgent
from rainy.envs import PyBullet, pybullet_parallel
from rainy.net.policy import SeparateStdGaussianDist


@rainy.main(A2CAgent, script_path=os.path.realpath(__file__))
def main(envname: str = "Hopper") -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: PyBullet(envname))
    c.set_net_fn(
        "actor-critic", rainy.net.actor_critic.fc_shared(policy=SeparateStdGaussianDist)
    )
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


if __name__ == "__main__":
    main()
