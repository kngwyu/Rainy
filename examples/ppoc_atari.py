"""Train PPOC agent in ALE game registerd in gym.
"""
import os

from torch.optim import RMSprop

import rainy
from rainy.agents import PPOCAgent
from rainy.envs import Atari, atari_parallel


@rainy.main(PPOCAgent, os.path.realpath(__file__))
def main(
    envname: str = "Breakout",
    num_options: int = 4,
    opt_delib_cost: float = 0.0,
    opt_beta_adv_merginal: float = 0.01,
    opt_avg_baseline: bool = False,
    proximal_update_for_mu: bool = False,
) -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: Atari(envname, frame_stack=False))
    c.set_parallel_env(atari_parallel())
    c.set_optimizer(lambda params: RMSprop(params, lr=7e-4, alpha=0.99, eps=1e-5))
    c.max_steps = int(2e7)
    c.grad_clip = 0.5
    # Option settings
    c.opt_delib_cost = opt_delib_cost
    c.opt_beta_adv_merginal = opt_beta_adv_merginal
    c.set_net_fn(
        "option-critic",
        rainy.net.option_critic.conv_shared(num_options=num_options, has_mu=True),
    )
    # PPO params
    c.nworkers = 8
    c.nsteps = 128
    c.value_loss_weight = 1.0
    c.gae_lambda = 0.95
    c.ppo_minibatch_size = 32 * 8
    c.ppo_clip = 0.1
    c.ppo_epochs = 3
    c.use_gae = True
    # Eval settings
    c.eval_env = Atari(envname)
    c.eval_deterministic = False
    c.episode_log_freq = 100
    c.eval_freq = c.max_steps // 10
    c.save_freq = None
    return c


if __name__ == "__main__":
    main()
