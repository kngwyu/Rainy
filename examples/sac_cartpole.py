import os
from rainy import Config
from rainy.agents import SACAgent
from rainy.envs import ClassicControl
import rainy.utils.cli as cli
from torch.optim import Adam


def config(envname: str = "CartPoleSwingUpContinuous-v0") -> Config:
    c = Config()
    c.set_env(lambda: ClassicControl(envname))
    c.max_steps = int(1e5)
    c.set_optimizer(lambda params: Adam(params, lr=3e-4), key="actor")
    c.set_optimizer(lambda params: Adam(params, lr=3e-4), key="critic")
    c.set_optimizer(lambda params: Adam(params, lr=3e-4), key="entropy")
    c.replay_size = int(1e5)
    c.replay_batch_size = 256
    c.train_start = int(1e4)
    c.use_reward_monitor = True
    c.eval_deterministic = True
    c.eval_freq = c.max_steps // 10
    c.sync_freq = 1
    c.grad_clip = None
    return c


if __name__ == "__main__":
    cli.run_cli(config, SACAgent, os.path.realpath(__file__))