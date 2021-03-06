import numpy as np
import pytest
import torch

import rainy
from rainy import agents, envs, lib, net, replay

try:
    import pybullet  # noqa

    HAS_PYBULLET = True
except ImportError:
    HAS_PYBULLET = False


class QValueHook(lib.hooks.EvalHook):
    def setup(self, config) -> None:
        self.q_values = []
        self.q_value_mean = config.device.zeros(config.action_dim)

    def step(self, _env, _action, transition, net_outputs) -> None:
        self.q_values.append(net_outputs["q_value"].squeeze())
        if transition.terminal:
            self.q_value_mean = torch.stack(self.q_values).mean(dim=0)
            self.q_values.clear()


@pytest.mark.parametrize(
    "make_ag, is_bootdqn", [(agents.DQNAgent, False), (agents.BootDQNAgent, True)]
)
def test_qvalue_hook(make_ag: callable, is_bootdqn: bool) -> None:
    c = rainy.Config()
    hook = QValueHook()
    c.eval_hooks.append(hook)
    if is_bootdqn:
        c.set_replay_buffer(
            lambda capacity: replay.UniformReplayBuffer(replay.BootDQNReplayFeed)
        )
    ag = make_ag(c)
    c.initialize_hooks()
    _ = ag.eval_episode()
    ag.close()
    assert len(hook.q_values) == 0
    assert tuple(hook.q_value_mean.shape) == (c.action_dim,)


def test_video_hook_atari() -> None:
    c = rainy.Config()
    c.eval_hooks.append(lib.hooks.VideoWriterHook(video_name="BreakoutVideo"))
    c.set_net_fn("dqn", net.value.dqn_conv())
    c.set_env(lambda: envs.Atari("Breakout"))
    c.eval_env = envs.Atari("Breakout")
    ag = agents.DQNAgent(c)
    c.initialize_hooks()
    _ = ag.eval_episode()
    ag.close()
    videopath = c.logger.logdir.joinpath("BreakoutVideo-0.avi")
    assert videopath.exists()


@pytest.mark.skipif(not HAS_PYBULLET, reason="PyBullet is an optional dependency")
def test_video_hook_pybullet() -> None:
    c = rainy.Config()
    c.eval_hooks.append(lib.hooks.VideoWriterHook(video_name="HopperVideo"))
    c.set_env(lambda: envs.PyBullet("Hopper"))
    c.set_explorer(lambda: lib.explore.GaussianNoise())
    c.set_explorer(lambda: lib.explore.Greedy(), key="eval")
    c.set_optimizer(lambda params: torch.optim.Adam(params, lr=1e-3), key="actor")
    c.set_optimizer(lambda params: torch.optim.Adam(params, lr=1e-3), key="critic")
    ag = agents.DDPGAgent(c)
    c.initialize_hooks()
    _ = ag.eval_episode()
    ag.close()
    videopath = c.logger.logdir.joinpath("HopperVideo-0.avi")
    assert videopath.exists()


def test_image_hook_atari() -> None:
    c = rainy.Config()
    hook = lib.hooks.ImageWriterHook(
        out_dir="/tmp/rainy-acvp/imagehook-atari", transpose=(2, 0, 1)
    )
    c.eval_hooks.append(hook)
    c.set_net_fn("dqn", net.value.dqn_conv())
    c.set_env(lambda: envs.Atari("Breakout"))
    c.eval_env = envs.Atari("Breakout")
    ag = agents.DQNAgent(c)
    c.initialize_hooks()
    _ = ag.eval_episode()
    ag.close()
    episodes = np.load(hook.out_dir.joinpath("ep1.npz"))
    assert episodes["states"][0].shape == (3, 210, 160)
    assert len(episodes["actions"].shape) == 1


@pytest.mark.skipif(not HAS_PYBULLET, reason="PyBullet is an optional dependency")
def test_state_hook_pybullet() -> None:
    c = rainy.Config()
    hook = lib.hooks.StateWriterHook(out_dir="/tmp/rainy-acvp/imagehook-pybullet")
    c.eval_hooks.append(hook)
    c.set_env(lambda: envs.PyBullet("Hopper"))
    c.set_explorer(lambda: lib.explore.GaussianNoise())
    c.set_explorer(lambda: lib.explore.Greedy(), key="eval")
    c.set_optimizer(lambda params: torch.optim.Adam(params, lr=1e-3), key="actor")
    c.set_optimizer(lambda params: torch.optim.Adam(params, lr=1e-3), key="critic")
    ag = agents.DDPGAgent(c)
    c.initialize_hooks()
    _ = ag.eval_episode()
    ag.close()
    episodes = np.load(hook.out_dir.joinpath("ep1.npz"))
    assert episodes["states"][0].shape == (15,)  # state space
    assert episodes["actions"][0].shape == (3,)  # action space
