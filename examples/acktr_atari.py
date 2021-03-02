import os

import rainy
from rainy.agents import ACKTRAgent
from rainy.envs import Atari, atari_parallel
from rainy.lib import kfac


@rainy.main(ACKTRAgent, script_path=os.path.realpath(__file__))
def main(
    envname: str = "Breakout",
    tau: float = 32 * 20 // 2,
    update_freq: int = 10,
) -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: Atari(envname, frame_stack=False))
    c.set_optimizer(kfac.default_sgd(eta_max=0.2))
    c.set_preconditioner(
        lambda net: kfac.KfacPreConditioner(
            net,
            tau=tau,
            update_freq=update_freq,
            norm_scaler=kfac.SquaredFisherScaler(eta_max=0.2, delta=0.001),
        )
    )
    c.set_net_fn("actor-critic", rainy.net.actor_critic.conv_shared())
    c.nworkers = 32
    c.nsteps = 20
    c.set_parallel_env(atari_parallel())
    c.value_loss_weight = 1.0
    c.use_gae = True
    c.lr_min = 0.0
    c.max_steps = int(2e7)
    c.eval_env = Atari(envname)
    c.eval_freq = None
    c.episode_log_freq = 100
    c.eval_deterministic = False
    return c


if __name__ == "__main__":
    main()
