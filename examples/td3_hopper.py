import os

from torch.optim import Adam

import rainy
from rainy.agents import TD3Agent
from rainy.lib import explore


@rainy.main(TD3Agent, os.path.realpath(__file__))
def main(
    envname: str = "Hopper",
    nworkers: int = 1,
    mujoco: bool = False,
) -> rainy.Config:
    c = rainy.Config()
    if mujoco:
        c.set_env(lambda: rainy.envs.Mujoco(envname))
    else:
        c.set_env(lambda: rainy.envs.PyBullet(envname))
    c.max_steps = int(1e6)
    c.set_optimizer(lambda params: Adam(params, lr=1e-3), key="actor")
    c.set_optimizer(lambda params: Adam(params, lr=1e-3), key="critic")
    c.replay_size = int(1e6)
    c.train_start = int(1e4)
    c.set_explorer(lambda: explore.GaussianNoise())
    c.set_explorer(lambda: explore.Greedy(), key="eval")
    c.set_explorer(
        lambda: explore.GaussianNoise(explore.DummyCooler(0.2), 0.5), key="target"
    )
    c.eval_deterministic = True
    c.eval_freq = c.max_steps // 10
    c.grad_clip = None
    c.nworkers = nworkers
    c.replay_batch_size = 100 * nworkers
    return c


if __name__ == "__main__":
    main()
