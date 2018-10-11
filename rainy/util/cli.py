import click
from typing import Tuple

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
def train(ctx: dict) -> None:
    c = ctx.obj['config']
    name = ctx.obj['name']
    c.logger.set_dir_from_prefix(name)
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
        r = ag.random_episode_and_save(fname)
    else:
        r = ag.random_episode(save)
    print("random play: {}".format(r))


@rainy_cli.command()
@click.option('--logdir', required=True, type=str)
@click.option('--fname', type=str, default='best-actions.json')
@click.pass_context
def eval(ctx: dict, logdir: str, fname: str) -> None:
    c = ctx.obj['config']
    ag = ctx.obj['make_agent'](c)
    eval_agent(ag, logdir, action_file=fname)


def run_cli(name: str, config: Config, agent_gen: Callable[[Config], Agent]) -> rainy_cli:
    ctx = {
        'name': name,
        'config': config,
        'agent': agent_gen,
    }
    rainy_cli(ctx)

