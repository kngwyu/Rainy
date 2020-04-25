import os

from torch.optim import Adam

import rainy
from rainy.agents import PPOCAgent
from rainy.envs import PyBullet, pybullet_parallel
from rainy.net.policy import SeparateStdGaussianDist


@rainy.main(PPOCAgent, os.path.realpath(__file__))
def main(
    envname: str = "Hopper",
    num_options: int = 2,
    opt_delib_cost: float = 0.0,
    opt_beta_adv_merginal: float = 0.01,
    opt_avg_baseline: bool = False,
    proximal_update_for_mu: bool = False,
) -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: PyBullet(envname))
    c.set_parallel_env(pybullet_parallel(normalize_obs=True, normalize_reward=True))
    c.set_optimizer(lambda params: Adam(params, lr=3.0e-4, eps=1.0e-4))
    c.max_steps = int(1e6)
    c.grad_clip = 0.5
    # Option settings
    c.opt_delib_cost = opt_delib_cost
    c.opt_beta_adv_merginal = opt_beta_adv_merginal
    c.set_net_fn(
        "option-critic",
        rainy.net.option_critic.fc_shared(
            num_options=num_options, policy=SeparateStdGaussianDist, has_mu=True
        ),
    )
    # PPO params
    c.nworkers = 4
    c.nsteps = 512
    c.ppo_minibatch_size = (4 * 512) // 8
    c.ppo_clip = 0.2
    c.use_gae = True
    c.eval_freq = c.max_steps // 10
    c.entropy_weight = 0.01
    c.value_loss_weight = 1.0
    c.eval_deterministic = True
    c.eval_times = 4
    c.proximal_update_for_mu = proximal_update_for_mu
    return c


if __name__ == "__main__":
    main()
