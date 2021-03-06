from os.path import realpath

import ppo_atari

import rainy
from rainy.envs import Atari, atari_parallel


@rainy.main(rainy.agents.PPOAgent, script_path=realpath(__file__))
def main(envname: str = "Breakout") -> rainy.Config:
    c = ppo_atari.main(envname)
    c.set_env(lambda: Atari(envname, flicker_frame=True, frame_stack=False))
    c.set_parallel_env(atari_parallel(frame_stack=False))
    c.set_net_fn(
        "actor-critic", rainy.net.actor_critic.conv_shared(rnn=rainy.net.GruBlock)
    )
    c.eval_env = Atari(envname, flicker_frame=True, frame_stack=True)
    return c


if __name__ == "__main__":
    main()
