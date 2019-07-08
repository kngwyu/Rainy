import click
from typing import Callable, Optional, Tuple

from .log import ExperimentLog
from ..agents import Agent
from ..config import Config
from ..lib import mpi
from .. import run


@click.group()
@click.option('--gpu', required=False, type=int, help='How many gpus you allow Rainy to use')
@click.option('--seed', type=int, default=None, help='Random seed set before training')
@click.option('--override', type=str, default='', help='Override string(See README for detail)')
@click.pass_context
def rainy_cli(ctx: dict, gpu: Tuple[int], seed: Optional[int], override: str) -> None:
    ctx.obj['gpu'] = gpu
    ctx.obj['config'].seed = seed
    if len(override) > 0:
        import builtins
        try:
            exec(override, builtins.__dict__, {'config': ctx.obj['config']})
        except Exception as e:
            print('!!! Your override string \'{}\' contains an error !!!'.format(override))
            raise e


@rainy_cli.command(help='Train agents')
@click.pass_context
@click.option('--comment', type=str, default=None, help='Comment wrote to fingerprint.txt')
@click.option('--prefix', type=str, default='', help='Prefix of the log directory')
def train(ctx: dict, comment: Optional[str], prefix: str) -> None:
    c = ctx.obj['config']
    scr = ctx.obj['script_path']
    if scr:
        c.logger.set_dir_from_script_path(scr, comment=comment, prefix=prefix)
    c.logger.set_stderr()
    ag = ctx.obj['make_agent'](c)
    run.train_agent(ag)
    if mpi.IS_MPI_ROOT:
        print("random play: {}, trained: {}".format(ag.random_episode(), ag.eval_episode()))


@rainy_cli.command(help='Run the random agent and show its result')
@click.option('--save', is_flag=True, help='Save actions')
@click.option('--render', is_flag=True, help='Render the agent')
@click.option('--replay', is_flag=True, help='Show replay(works with special environments)')
@click.option('--action-file', type=str,
              default='best-actions.json', help='Name of the action file')
@click.pass_context
def random(ctx: dict, save: bool, render: bool, replay: bool, action_file: str) -> None:
    c = ctx.obj['config']
    ag = ctx.obj['make_agent'](c)
    action_file = fname if save else None
    run.random_agent(ag, render=render, replay=replay, action_file=action_file)


@rainy_cli.command(help='Load a save file and restart training')
@click.pass_context
@click.argument('logdir')
@click.option('--model', type=str, default=run.SAVE_FILE_DEFAULT, help='Name of the save file')
@click.option('--additional-steps', type=int,
              default=100, help='The number of  additional training steps')
def retrain(ctx: dict, logdir: str, model: str, additional_steps: int) -> None:
    c = ctx.obj['config']
    log = c.logger.retrive(logdir)
    c.logger.set_stderr()
    ag = ctx.obj['make_agent'](c)
    run.retrain_agent(ag, log, load_file_name=model, additional_steps=additional_steps)
    print("random play: {}, trained: {}".format(ag.random_episode(), ag.eval_episode()))


@rainy_cli.command(help='Load a save file and evaluate the agent')
@click.argument('logdir')
@click.option('--model', type=str, default=run.SAVE_FILE_DEFAULT, help='Name of the save file')
@click.option('--render', is_flag=True, help='Render the agent')
@click.option('--replay', is_flag=True, help='Show replay(works with special environments)')
@click.option('--action-file', type=str,
              default='best-actions.json', help='Name of the action file')
@click.pass_context
def eval(ctx: dict, logdir: str, model: str, render: bool, replay: bool, action_file: str) -> None:
    c = ctx.obj['config']
    ag = ctx.obj['make_agent'](c)
    run.eval_agent(
        ag,
        logdir,
        load_file_name=model,
        render=render,
        replay=replay,
        action_file=action_file
    )


@rainy_cli.command(help='Open an ipython shell with rainy imported')
@click.option('--logdir', type=str, help='Name of the directly where the log file')
@click.option('--vi-mode', is_flag=True, help='Open ipython shell with vi-mode enabled')
@click.pass_context
def ipython(ctx: dict, logdir: Optional[str], vi_mode: bool) -> None:
    config, make_agent = ctx.obj['config'], ctx.obj['make_agent']  # noqa
    if logdir is not None:
        log = ExperimentLog(logdir)  # noqa
    else:
        open_log = ExperimentLog  # noqa
    try:
        from ptpython.ipython import embed
        del ctx, logdir
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

