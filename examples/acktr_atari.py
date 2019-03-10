import os
from rainy import Config, net
from rainy.agents import AcktrAgent
from rainy.envs import Atari, atari_parallel
import rainy.utils.cli as cli
from rainy.lib import kfac

KFAC_KWARGS = {
    'tau': 32 * 20 // 2,
    'update_freq': 10,
    'norm_scaler': kfac.SquaredFisherScaler(eta_max=0.2, delta=0.001),
}


def config() -> Config:
    c = Config()
    c.set_env(lambda: Atari('Breakout', frame_stack=False))
    c.set_optimizer(kfac.default_sgd(eta_max=0.2))
    c.set_preconditioner(lambda net: kfac.KfacPreConditioner(net, **KFAC_KWARGS))
    c.set_net_fn('actor-critic', net.actor_critic.ac_conv())
    c.nworkers = 32
    c.nsteps = 20
    c.set_parallel_env(atari_parallel())
    c.value_loss_weight = 0.5
    c.use_gae = True
    c.lr_decay = True
    c.max_steps = int(2e7)
    c.eval_env = Atari('Breakout')
    c.eval_freq = None
    c.episode_log_freq = 100
    c.use_reward_monitor = True
    c.eval_deterministic = False
    return c


if __name__ == '__main__':
    cli.run_cli(config(), AcktrAgent, script_path=os.path.realpath(__file__))
