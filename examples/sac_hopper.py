import os

from torch.optim import Adam

import rainy
from rainy.agents import SACAgent


@rainy.main(SACAgent, os.path.realpath(__file__))
def main(
    envname: str = "Hopper", nworkers: int = 1, mujoco: bool = False,
) -> rainy.Config:
    c = rainy.Config()
    if mujoco:
        c.set_env(lambda: rainy.envs.Mujoco(envname))
    else:
        c.set_env(lambda: rainy.envs.PyBullet(envname))
    c.max_steps = int(1e6)
    c.set_optimizer(lambda params: Adam(params, lr=3e-4), key="actor")
    c.set_optimizer(lambda params: Adam(params, lr=3e-4), key="critic")
    c.set_optimizer(lambda params: Adam(params, lr=3e-4), key="entropy")
    c.replay_size = int(1e6)
    c.train_start = int(1e4)
    c.eval_deterministic = True
    c.eval_freq = c.max_steps // 100
    c.sync_freq = 1
    c.grad_clip = None
    c.nworkers = nworkers
    c.replay_batch_size = 256 * nworkers
    return c


if __name__ == "__main__":
    main()
