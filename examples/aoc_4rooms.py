"""
This example needs rlpy3, which you can install by::

  pip3 install rlpy3 -U --pre
"""
import os

import click
import torch
from torch import optim

import rainy
from rainy.envs import ClassicControl, MultiProcEnv
from rainy.lib.hooks import EvalHook
from rainy.net import option_critic as oc
from rainy.prelude import State
from rainy.utils.cli import run_cli
from rlpy.gym import RLPyEnv


class OptionVisualizeHook(EvalHook):
    def __init__(
        self, num_options: int, vis_beta: bool = True, vis_pi: bool = True,
    ) -> None:
        self.num_options = num_options
        self.vis_beta = vis_beta
        self.vis_pi = vis_pi
        self.initial = True

    def setup(self, config: rainy.Config) -> None:
        self.device = config.device

    def reset(
        self, agent: rainy.agents.Agent, env: rainy.envs.EnvExt, initial_state: State
    ) -> None:
        states = self._all_states(env.unwrapped, initial_state)
        if self.vis_beta or self.vis_p:
            with torch.no_grad():
                pi, q, beta = agent.net(states)

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

        if self.vis_pi:
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

    def _all_states(self, env: RLPyEnv, initial_state: State) -> torch.Tensor:
        s = []
        for state in env.domain.all_states():
            s.append(self.device.tensor(env.get_obs(state)))
        return torch.stack(s)


def config(
    envname: str = "RLPyGridWorld11x11-4Rooms-RandomGoal-v2",
    num_options: int = 4,
    opt_delib_cost: float = 0.0,
    opt_beta_adv_merginal: float = 0.01,
    use_gae: bool = False,
    opt_avg_baseline: bool = False,
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
    c.grad_clip = 1.0
    c.eval_freq = c.max_steps // 20
    c.network_log_freq = (c.max_steps // c.batch_size) // 10
    c.entropy_weight = 0.001
    c.value_loss_weight = 1.0
    c.opt_delib_cost = opt_delib_cost
    c.opt_beta_adv_merginal = opt_beta_adv_merginal
    c.opt_avg_baseline = opt_avg_baseline
    c.use_gae = use_gae
    if "v2" in envname:
        c.set_net_fn(
            "option-critic",
            oc.conv_shared(
                num_options=num_options,
                hidden_channels=(8, 8),
                output_dim=128,
                cnn_params=[(4, 1), (2, 1)],
            ),
        )
    else:
        c.set_net_fn("option-critic", oc.fc_shared(num_options=num_options))
    return c


if __name__ == "__main__":
    options = [
        click.Option(["--num-options"], type=int, default=4),
        click.Option(["--opt-delib-cost"], type=float, default=0.0),
        click.Option(["--opt-beta-adv-merginal"], type=float, default=0.01),
        click.Option(["--use-gae"], is_flag=True),
        click.Option(["--opt-avg-baseline"], is_flag=True),
        click.Option(["--visualize-beta", "-VB"], is_flag=True),
    ]
    run_cli(config, rainy.agents.AOCAgent, os.path.realpath(__file__), options)
