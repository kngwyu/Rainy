from os.path import realpath
import ppo_atari
import rainy
from rainy.envs import Atari, atari_parallel
import rainy.utils.cli as cli


def config() -> rainy.Config:
    c = ppo_atari.config()
    c.set_env(lambda: Atari('Breakout', flicker_frame=True, frame_stack=False))
    c.set_parallel_env(atari_parallel(frame_stack=False))
    c.set_net_fn('actor-critic', rainy.net.actor_critic.ac_conv(rnn=rainy.net.GruBlock))
    c.eval_env = Atari('Breakout', flicker_frame=True, frame_stack=True)
    return c


if __name__ == '__main__':
    cli.run_cli(config(), rainy.agents.PpoAgent, script_path=realpath(__file__))
