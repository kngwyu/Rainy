import os

from torch.optim import Adam

import rainy
from rainy.envs import Atari, atari_parallel


@rainy.main(rainy.agents.PPOAgent, script_path=os.path.realpath(__file__))
def main(envname: str = "Breakout", use_rnn: bool = False,) -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: Atari(envname, frame_stack=False))
    if use_rnn:
        c.set_net_fn(
            "actor-critic", rainy.net.actor_critic.conv_shared(rnn=net.GruBlock)
        )
    else:
        c.set_net_fn("actor-critic", rainy.net.actor_critic.conv_shared())
    c.set_parallel_env(atari_parallel())
    c.set_optimizer(lambda params: Adam(params, lr=2.5e-4, eps=1.0e-4))
    c.max_steps = int(2e7)
    c.grad_clip = 0.5
    # ppo params
    c.nworkers = 8
    c.nsteps = 128
    c.value_loss_weight = 1.0
    c.gae_lambda = 0.95
    c.ppo_minibatch_size = 32 * 8
    c.ppo_clip = 0.1
    c.ppo_epochs = 3
    c.use_gae = True
    c.lr_min = None  # set 0.0 if you decrease ppo_clip
    # eval settings
    c.eval_env = Atari(envname)
    c.episode_log_freq = 100
    c.eval_freq = None
    c.save_freq = None
    return c


if __name__ == "__main__":
    main()
