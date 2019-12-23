import os
from rainy import Config
from rainy.agents import TD3Agent
from rainy.envs import ClassicControl
from rainy.lib import explore
import rainy.utils.cli as cli
from torch.optim import Adam


def config(envname: str = "CartPoleSwingUpContinuous-v0") -> Config:
    c = Config()
    c.set_env(lambda: ClassicControl(envname))
    c.max_steps = int(1e5)
    c.set_optimizer(lambda params: Adam(params, lr=1e-3), key="actor")
    c.set_optimizer(lambda params: Adam(params, lr=1e-3), key="critic")
    c.replay_size = int(1e5)
    c.replay_batch_size = 100
    c.train_start = int(1e3)
    c.set_explorer(lambda: explore.GaussianNoise())
    c.set_explorer(lambda: explore.Greedy(), key="eval")
    c.set_explorer(
        lambda: explore.GaussianNoise(explore.DummyCooler(0.2), 0.5), key="target"
    )
    c.use_reward_monitor = True
    c.eval_deterministic = True
    c.eval_freq = c.max_steps // 10
    c.grad_clip = None
    return c


if __name__ == "__main__":
    cli.run_cli(config, TD3Agent, os.path.realpath(__file__))