import click
from typing import Callable, Optional, Tuple

from ..agents import Agent
from ..config import Config
from ..ipython import _open_ipython
from ..lib import mpi
from .. import run


@click.group()
@click.option(
    "--gpu", required=False, type=int, help="How many gpus you allow the script to use"
)
@click.option(
    "--envname", type=str, default=None, help="Name of environment passed to config_gen"
)
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed set before training. Left for backward comaptibility",
)
@click.option(
    "--override", type=str, default="", help="Override string(see README for detail)"
)
@click.pass_context
def rainy_cli(
    ctx: click.Context,
    gpu: Tuple[int],
    envname: Optional[str],
    seed: Optional[int],
    override: str,
) -> None:
    ctx.obj["gpu"] = gpu
    cfg_gen = ctx.obj["config_gen"]
    ctx.obj["config"] = cfg_gen(envname) if envname is not None else cfg_gen()
    ctx.obj["config"].seed = seed
    ctx.obj["override"] = override
    ctx.obj["envname"] = "Default" if envname is None else envname
    if len(override) > 0:
        import builtins

        try:
            exec(override, builtins.__dict__, {"config": ctx.obj["config"]})
        except Exception as e:
            print(
                "!!! Your override string '{}' contains an error !!!".format(override)
            )
            raise e


@rainy_cli.command(help="Train agents")
@click.pass_context
@click.option(
    "--comment",
    type=str,
    default=None,
    help="Comment that would be wrote to fingerprint.txt",
)
@click.option("--prefix", type=str, default="", help="Prefix of the log directory")
def train(ctx: click.Context, comment: Optional[str], prefix: str) -> None:
    c = ctx.obj["config"]
    script_path = ctx.obj["script_path"]
    if script_path is not None:
        fingerprint = dict(
            comment="" if comment is None else comment,
            envname=ctx.obj["envname"],
            override=ctx.obj["override"],
        )
        c.logger.set_dir_from_script_path(
            script_path, prefix=prefix, fingerprint=fingerprint
        )
    ag = ctx.obj["make_agent"](c)
    run.train_agent(ag)
    if mpi.IS_MPI_ROOT:
        print(
            "random play: {}, trained: {}".format(
                ag.random_episode(), ag.eval_episode()
            )
        )


@rainy_cli.command(help="Run the random agent and show its result")
@click.option("--save", is_flag=True, help="Save actions")
@click.option("--render", is_flag=True, help="Render the agent")
@click.option(
    "--replay",
    is_flag=True,
    help="Show replay(works only with special environments, e.g., rogue-gym)",
)
@click.option(
    "--action-file",
    type=str,
    default="best-actions.json",
    help="Name of the action file",
)
@click.pass_context
def random(
    ctx: click.Context, save: bool, render: bool, replay: bool, action_file: str
) -> None:
    c = ctx.obj["config"]
    ag = ctx.obj["make_agent"](c)
    if save:
        run.random_agent(ag, render=render, replay=replay, action_file=action_file)
    else:
        run.random_agent(ag, render=render, replay=replay)


@rainy_cli.command(help="Given a save file and restart training")
@click.pass_context
@click.argument("logdir", type=str)
@click.option(
    "--model", type=str, default=run.SAVE_FILE_DEFAULT, help="Name of the save file"
)
@click.option(
    "--additional-steps",
    type=int,
    default=100,
    help="The number of  additional training steps",
)
def retrain(ctx: click.Context, logdir: str, model: str, additional_steps: int) -> None:
    c = ctx.obj["config"]
    ag = ctx.obj["make_agent"](c)
    run.retrain_agent(
        ag, logdir, load_file_name=model, additional_steps=additional_steps
    )
    print("random play: {}, trained: {}".format(ag.random_episode(), ag.eval_episode()))


@rainy_cli.command(help="Load a specified save file and evaluate the agent")
@click.argument("logdir", type=str)
@click.option(
    "--model", type=str, default=run.SAVE_FILE_DEFAULT, help="Name of the save file"
)
@click.option("--render", is_flag=True, help="Render the agent")
@click.option(
    "--replay",
    is_flag=True,
    help="Show replay(works only with special environments, e.g., rogue-gym)",
)
@click.option(
    "--action-file",
    type=str,
    default="best-actions.json",
    help="Name of the action file",
)
@click.pass_context
def eval(
    ctx: click.Context,
    logdir: str,
    model: str,
    render: bool,
    replay: bool,
    action_file: str,
) -> None:
    c = ctx.obj["config"]
    ag = ctx.obj["make_agent"](c)
    run.eval_agent(
        ag,
        logdir,
        load_file_name=model,
        render=render,
        replay=replay,
        action_file=action_file,
    )


@rainy_cli.command(help="Open an ipython shell with rainy imported")
@click.option("--logdir", type=str, help="Name of the directly where the log file")
@click.pass_context
def ipython(ctx: click.Context, logdir: Optional[str]) -> None:
    config, make_agent = ctx.obj["config"], ctx.obj["make_agent"]  # noqa
    _open_ipython(logdir)


def run_cli(
    config_gen: Callable[..., Config],
    agent_gen: Callable[[Config], Agent],
    script_path: Optional[str] = None,
) -> None:
    obj = {
        "config_gen": config_gen,
        "make_agent": agent_gen,
        "script_path": script_path,
    }
    rainy_cli(obj=obj)
