"""Train A2C agent in ALE game registerd in gym.
Some hyper parametes are from OpenAI baselines:
https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py
"""
import os

from torch.optim import RMSprop

import rainy
from rainy.agents import A2CAgent
from rainy.envs import Atari, atari_parallel


@rainy.main(A2CAgent, script_path=os.path.realpath(__file__))
def main(envname: str = "Breakout") -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: Atari(envname, frame_stack=False))
    c.set_optimizer(lambda params: RMSprop(params, lr=7e-4, alpha=0.99, eps=1e-5))
    # c.set_net_fn('actor-critic', rainy.net.actor_critic.conv_shared(rnn=net.GruBlock))
    c.set_net_fn("actor-critic", rainy.net.actor_critic.conv_shared())
    c.nworkers = 16
    c.nsteps = 5
    c.set_parallel_env(atari_parallel())
    c.grad_clip = 0.5
    c.value_loss_weight = 1.0
    c.use_gae = False
    c.max_steps = int(2e7)
    c.eval_env = Atari(envname)
    c.eval_deterministic = False
    c.episode_log_freq = 100
    c.eval_freq = None
    c.save_freq = None
    return c


if __name__ == "__main__":
    main()
