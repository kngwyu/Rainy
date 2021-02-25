import os

import rainy
from rainy.agents import ACKTRAgent
from rainy.envs import ClassicControl, MultiProcEnv
from rainy.lib import kfac


@rainy.main(ACKTRAgent, script_path=os.path.realpath(__file__))
def main(
    envname: str = "CartPole-v0",
    tau: float = 12 * 20,
    update_freq: int = 10,
) -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: ClassicControl(envname))
    c.max_steps = int(4e5)
    c.nworkers = 12
    c.nsteps = 20
    c.set_parallel_env(MultiProcEnv)
    c.set_optimizer(kfac.default_sgd(eta_max=0.1))
    c.set_preconditioner(
        lambda net: kfac.KfacPreConditioner(
            net,
            tau=tau,
            update_freq=update_freq,
            norm_scaler=kfac.SquaredFisherScaler(eta_max=0.1, delta=0.001),
        )
    )
    c.gae_lambda = 0.95
    c.use_gae = False
    c.lr_min = 0.0
    c.value_loss_weight = 0.2
    c.entropy_weight = 0.01
    c.eval_freq = None
    return c


if __name__ == "__main__":
    main()
