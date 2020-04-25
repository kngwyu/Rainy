import os

from torch import optim

import rainy
from rainy.envs import ClassicControl, MultiProcEnv


@rainy.main(rainy.agents.PPOCAgent, os.path.realpath(__file__))
def main(
    envname: str = "CartPole-v0",
    num_options: int = 2,
    opt_delib_cost: float = 0.0,
    opt_beta_adv_merginal: float = 0.01,
    opt_avg_baseline: bool = False,
    proximal_update_for_mu: bool = False,
) -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: ClassicControl(envname))
    c.max_steps = int(4e5)
    # Option settings
    c.opt_delib_cost = opt_delib_cost
    c.opt_beta_adv_merginal = opt_beta_adv_merginal
    c.set_net_fn(
        "option-critic",
        rainy.net.option_critic.fc_shared(num_options=num_options, has_mu=True),
    )
    # PPO params
    c.nworkers = 12
    c.nsteps = 64
    c.set_parallel_env(MultiProcEnv)
    c.set_optimizer(lambda params: optim.Adam(params, lr=2.5e-4, eps=1.0e-4))
    c.grad_clip = 1.0
    c.eval_freq = 10000
    c.entropy_weight = 0.01
    c.value_loss_weight = 1.0
    c.use_gae = True
    return c


if __name__ == "__main__":
    main()
