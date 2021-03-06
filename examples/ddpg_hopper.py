import os

from torch.optim import Adam

import rainy
from rainy import Config
from rainy.agents import DDPGAgent
from rainy.envs import PyBullet
from rainy.lib import explore


@rainy.main(DDPGAgent, os.path.realpath(__file__))
def main(envname: str = "Hopper", nworkers: int = 1) -> Config:
    c = Config()
    c.set_env(lambda: PyBullet(envname))
    c.max_steps = int(1e6)
    c.set_optimizer(lambda params: Adam(params, lr=1e-3), key="actor")
    c.set_optimizer(lambda params: Adam(params, lr=1e-3), key="critic")
    c.replay_size = int(1e6)
    c.train_start = int(1e4)
    c.set_explorer(lambda: explore.GaussianNoise())
    c.set_explorer(lambda: explore.Greedy(), key="eval")
    c.eval_deterministic = True
    c.grad_clip = None
    c.eval_freq = c.max_steps // 10
    c.nworkers = nworkers
    c.replay_batch_size = 100 * nworkers
    return c


if __name__ == "__main__":
    main()
