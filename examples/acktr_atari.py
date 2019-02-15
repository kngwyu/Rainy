import os
from rainy import Config, net
from rainy.agents import AcktrAgent
from rainy.envs import Atari
import rainy.utils.cli as cli
from rainy.envs import FrameStackParallel, MultiProcEnv
from rainy.lib import kfac

KFAC_KWARGS = {
    'eta_max': 0.2,
    'tau': 32 * 20,
}


def config() -> Config:
    c = Config()
    c.set_env(lambda: Atari('Breakout', frame_stack=False))
    c.set_optimizer(kfac.default_sgd(KFAC_KWARGS['eta_max']))
    c.set_preconditioner(lambda net: kfac.KFAC(net, eps=0.01, pi=True, constraint_norm=True))
    c.set_net_fn('actor-critic', net.actor_critic.ac_conv)
    c.nworkers = 32
    c.nsteps = 20
    c.set_parallel_env(lambda env_gen, num_w: FrameStackParallel(MultiProcEnv(env_gen, num_w)))
    c.value_loss_weight = 0.5
    c.use_gae = True
    c.lr_decay = False
    c.max_steps = int(2e7)
    c.eval_env = Atari('Breakout')
    c.eval_freq = None
    c.episode_log_freq = 100
    c.use_reward_monitor = True
    c.eval_deterministic = False
    return c


if __name__ == '__main__':
    cli.run_cli(config(), AcktrAgent, script_path=os.path.realpath(__file__))
