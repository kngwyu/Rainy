"""
This example needs rlpy3, which you can install by::

  pip3 install rlpy3 -U --pre
"""
import click
import os
import rainy
from rainy.envs import ClassicControl, MultiProcEnv
from rainy.lib.hooks import EvalHook
from rainy.net import termination_critic as tc
from rainy.prelude import State
from rainy.utils.cli import run_cli
import torch
from torch import optim, Tensor
from typing import Tuple
from rlpy.gym import RLPyEnv


class OptionVisualizeHook(EvalHook):
    def __init__(
        self,
        num_options: int,
        vis_beta: bool = True,
        vis_p: bool = False,
        vis_pi: bool = True,
    ) -> None:
        self.num_options = num_options
        self.vis_beta = vis_beta
        self.vis_p = vis_p
        self.vis_pi = vis_pi
        self.initial = True

    def setup(self, config: rainy.Config) -> None:
        self.device = config.device

    def reset(
        self, agent: rainy.agents.Agent, env: rainy.envs.EnvExt, initial_state: State
    ) -> None:
        xs, xf = self._xs_xf(env.unwrapped, initial_state)
        if self.vis_beta or self.vis_p:
            with torch.no_grad():
                beta, p, _, _ = agent.tc_net(xs, xf)

        if self.vis_beta:
            beta = beta.dist.probs.cpu().numpy()
            for i in range(self.num_options):
                env.unwrapped.domain.show_heatmap(
                    beta[:, i],
                    "β(Xf)",
                    normalize_method="uniform",
                    cmap="PuOr",
                    nrows=2,
                    ncols=2,
                    index=i + 1,
                    ticks=False,
                    title=f"Option: {i}",
                    legend=self.initial and i == 1,
                )

        if self.vis_p:
            p = p.cpu().numpy()
            for i in range(self.num_options):
                env.unwrapped.domain.show_heatmap(
                    p[:, i],
                    "P(Xs|Xf)",
                    normalize_method="uniform",
                    cmap="PuOr",
                    title=f"Option: {i}",
                    nrows=2,
                    ncols=2,
                    index=i + 1,
                )

        if self.vis_pi:
            with torch.no_grad():
                pi, q = agent.ac_net(xf)
                pi = pi.dist.probs.cpu().numpy()
                q = q.cpu().numpy()
            for i in range(self.num_options):
                env.unwrapped.domain.show_policy(
                    pi[:, i, :],
                    q[:, i],
                    nrows=2,
                    ncols=2,
                    index=i + 1,
                    scale=1.6,
                    ticks=False,
                    title=f"Option: {i}",
                )

        self.initial = True

    def _xs_xf(self, env: RLPyEnv, initial_state: State) -> Tuple[Tensor, Tensor]:
        xf = []
        for state in env.domain.all_states():
            xf.append(self.device.tensor(env.get_obs(state)))
        xs = torch.stack([self.device.tensor(initial_state) for _ in range(len(xf))])
        return xs, torch.stack(xf)


def config(
    envname: str = "RLPyGridWorld11x11-4Rooms-RandomGoal-v2",
    num_options: int = 4,
    visualize_beta: bool = False,
) -> rainy.Config:
    c = rainy.Config()
    if visualize_beta:
        c.eval_hooks.append(OptionVisualizeHook(num_options))
    c.set_env(lambda: ClassicControl(envname))
    c.max_steps = int(4e5)
    c.nworkers = 12
    c.nsteps = 5
    c.set_parallel_env(MultiProcEnv)
    c.set_optimizer(lambda params: optim.RMSprop(params, lr=0.0007))
    c.set_optimizer(lambda params: optim.Adam(params, lr=1e-4), key="termination")
    c.set_explorer(lambda: rainy.lib.explore.EpsGreedy(0.1))
    c.grad_clip = 1.0
    c.eval_freq = c.max_steps // 20
    c.network_log_freq = (c.max_steps // c.batch_size) // 10
    c.entropy_weight = 0.001
    c.value_loss_weight = 1.0
    if "v2" in envname:
        CONV_ARGS = dict(
            hidden_channels=(8, 8),
            feature_dim=128,
            kernel_and_strides=[(4, 1), (2, 1)],
        )
        c.set_net_fn(
            "actor-critic", tc.oac_conv_shared(num_options=num_options, **CONV_ARGS),
        )
        c.set_net_fn(
            "termination-critic",
            tc.tc_conv_shared(num_options=num_options, **CONV_ARGS),
        )
    else:
        c.set_net_fn(
            "actor-critic", tc.oac_fc_shared(num_options=num_options),
        )
        c.set_net_fn(
            "termination-critic", tc.tc_fc_shared(num_options=num_options),
        )
    return c


if __name__ == "__main__":
    options = [
        click.Option(["--num-options"], type=int, default=4),
        click.Option(["--visualize-beta", "-VB"], is_flag=True),
    ]
    run_cli(config, rainy.agents.ACTCAgent, os.path.realpath(__file__), options)
