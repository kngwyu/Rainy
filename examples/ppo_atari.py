import os
from rainy import Config, net
from rainy.agents import PpoAgent
import rainy.utils.cli as cli
from rainy.envs import Atari, atari_parallel
from torch.optim import Adam


def config() -> Config:
    c = Config()
    c.set_env(lambda: Atari('Breakout', frame_stack=False))
    c.set_net_fn('actor-critic', net.actor_critic.ac_conv())
    c.set_parallel_env(atari_parallel())
    c.set_optimizer(lambda params: Adam(params, lr=2.5e-4, eps=1.0e-4))
    c.max_steps = int(2e7)
    c.grad_clip = 0.5
    # ppo params
    c.nworkers = 8
    c.nsteps = 128
    c.value_loss_weight = 0.5
    c.gae_tau = 0.95
    c.ppo_minibatch_size = (128 * 8) // 4
    c.ppo_clip = 0.1
    c.use_gae = True
    c.use_reward_monitor = True
    c.lr_decay = True
    # eval settings
    c.eval_env = Atari('Breakout')
    c.eval_freq = None
    c.episode_log_freq = 100
    return c


if __name__ == '__main__':
    cli.run_cli(config(), PpoAgent, script_path=os.path.realpath(__file__))
