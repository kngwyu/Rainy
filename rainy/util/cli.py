import click
from typing import Callable, Optional, Tuple

from .log import ExperimentLog
from ..agent import Agent
from ..config import Config
from ..run import eval_agent, train_agent


@click.group()
@click.option('--gpu', required=False, type=int)
@click.pass_context
def rainy_cli(ctx: dict, gpu: Tuple[int]) -> None:
    ctx.obj['gpu'] = gpu


@rainy_cli.command()
@click.pass_context
@click.option('--comment', type=str, default=None)
@click.option('--prefix', type=str, default='')
@click.option('--seed', type=int, default=None)
def train(ctx: dict, comment: Optional[str], prefix: str, seed: Optional[int]) -> None:
    c = ctx.obj['config']
    scr = ctx.obj['script_path']
    if scr:
        c.logger.set_dir_from_script_path(scr, comment=comment, prefix=prefix)
    c.logger.set_stderr()
    c.seed = seed
    ag = ctx.obj['make_agent'](c)
    train_agent(ag)
    print("random play: {}, trained: {}".format(ag.random_episode(), ag.eval_episode()))


@rainy_cli.command()
@click.option('--save', is_flag=True)
@click.option('--fname', type=str, default='random-actions.json')
@click.pass_context
def random(ctx: dict, save: bool, fname: str) -> None:
    c = ctx.obj['config']
    ag = ctx.obj['make_agent'](c)
    if save:
        r = ag.random_and_save(fname)
    else:
        r = ag.random_episode()
    print("random play: {}".format(r))


@rainy_cli.command()
@click.argument('logdir', required=True, type=str)
@click.option('--render', is_flag=True)
@click.option('--replay', is_flag=True)
@click.option('--fname', type=str, default='best-actions.json')
@click.pass_context
def eval(ctx: dict, logdir: str, render: bool, replay: bool, fname: str) -> None:
    c = ctx.obj['config']
    ag = ctx.obj['make_agent'](c)
    eval_agent(ag, logdir, render=render, replay=replay, action_file=fname)


@rainy_cli.command()
@click.option('--log-dir', type=str)
@click.option('--vi-mode', is_flag=True)
@click.pass_context
def ipython(ctx: dict, log_dir: Optional[str], vi_mode: bool) -> None:
    config, make_agent = ctx.obj['config'], ctx.obj['make_agent']  # noqa
    if log_dir is not None:
        log = ExperimentLog(log_dir)  # noqa
    else:
        open_log = ExperimentLog  # noqa
    try:
        from ptpython.ipython import embed
        del ctx, log_dir
        import rainy  # noqa
        embed(vi_mode=vi_mode)
    except ImportError:
        print("To use ipython mode, install ipython and ptpython first.")


def run_cli(
        config: Config,
        agent_gen: Callable[[Config], Agent],
        script_path: Optional[str] = None
) -> rainy_cli:
    obj = {
        'config': config,
        'make_agent': agent_gen,
        'script_path': script_path
    }
    rainy_cli(obj=obj)

