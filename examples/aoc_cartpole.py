import os

from torch import optim

import rainy
from rainy.envs import ClassicControl, MultiProcEnv


@rainy.main(rainy.agents.AOCAgent, os.path.realpath(__file__))
def main(
    envname: str = "CartPole-v0",
    num_options: int = 2,
    opt_delib_cost: float = 0.0,
    opt_beta_adv_merginal: float = 0.01,
    use_gae: bool = False,
    opt_avg_baseline: bool = False,
) -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: ClassicControl(envname))
    c.max_steps = int(4e5)
    c.nworkers = 12
    c.nsteps = 5
    c.set_parallel_env(MultiProcEnv)
    c.set_optimizer(lambda params: optim.RMSprop(params, lr=0.0007))
    c.grad_clip = 1.0
    c.eval_freq = c.max_steps // 10
    c.network_log_freq = (c.max_steps // c.batch_size) // 10
    c.entropy_weight = 0.001
    c.value_loss_weight = 1.0
    c.opt_delib_cost = opt_delib_cost
    c.opt_beta_adv_merginal = opt_beta_adv_merginal
    c.opt_avg_baseline = opt_avg_baseline
    c.use_gae = use_gae
    c.set_net_fn(
        "option-critic", rainy.net.option_critic.fc_shared(num_options=num_options)
    )
    return c


if __name__ == "__main__":
    main()
