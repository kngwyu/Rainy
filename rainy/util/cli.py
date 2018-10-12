import click
from typing import Callable, Optional, Tuple

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
def train(ctx: dict, comment: Optional[str]) -> None:
    c = ctx.obj['config']
    scr = ctx.obj['script_path']
    if scr:
        c.logger.set_dir_from_script_path(scr, comment=comment)
    c.logger.set_stderr()
    ag = ctx.obj['make_agent'](c)
    train_agent(ag)
    print("random play: {}, trained: {}".format(ag.random_episode(), ag.eval_episode()))


@rainy_cli.command()
@click.option('--save', default=False)
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
@click.option('--fname', type=str, default='best-actions.json')
@click.pass_context
def eval(ctx: dict, logdir: str, fname: str) -> None:
    c = ctx.obj['config']
    ag = ctx.obj['make_agent'](c)
    eval_agent(ag, logdir, action_file=fname)


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

