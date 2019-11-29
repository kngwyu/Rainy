import click
from typing import Callable, List, Optional

from ..agents import Agent
from ..config import Config
from ..experiment import Experiment
from ..lib import mpi


@click.group()
@click.option(
    "--envname", type=str, default=None, help="Name of environment passed to config_gen"
)
@click.option("--max-steps", type=int, default=None, help="Max steps of the training")
@click.option(
    "--seed",
    type=int,
    default=None,
    help="Random seed set before training. Left for backward comaptibility",
)
@click.option("--model", type=str, default=None, help="Name of the save file")
@click.option(
    "--action-file", type=str, default="actions.json", help="Name of the action file",
)
@click.pass_context
def rainy_cli(
    ctx: click.Context,
    envname: Optional[str],
    max_steps: Optional[int],
    seed: Optional[int],
    model: Optional[str],
    action_file: Optional[str],
    **kwargs,
) -> None:
    cfg_gen = ctx.obj["config_gen"]
    if envname is not None:
        kwargs["envname"] = envname
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    config = cfg_gen(**kwargs)
    config.seed = seed
    ag = ctx.obj["agent_gen"](config)
    ctx.obj["experiment"] = Experiment(ag, model, action_file)
    ctx.obj["envname"] = "Default" if envname is None else envname
    ctx.obj["kwargs"] = kwargs


@rainy_cli.command(help="Train agents")
@click.pass_context
@click.option(
    "--comment",
    type=str,
    default=None,
    help="Comment that would be wrote to fingerprint.txt",
)
@click.option("--prefix", type=str, default="", help="Prefix of the log directory")
@click.option(
    "--eval-render", is_flag=True, help="Render the environment when evaluating"
)
def train(
    ctx: click.Context, comment: Optional[str], prefix: str, eval_render: bool = True
) -> None:
    script_path = ctx.obj["script_path"]
    experiment = ctx.obj["experiment"]
    if script_path is not None:
        fingerprint = dict(
            comment="" if comment is None else comment,
            envname=ctx.obj["envname"],
            kwargs=ctx.obj["kwargs"],
        )
        experiment.logger.setup_from_script_path(
            script_path, prefix=prefix, fingerprint=fingerprint
        )
    experiment.train(eval_render=eval_render)
    if mpi.IS_MPI_ROOT:
        print(
            "random play: {}, trained: {}".format(
                experiment.ag.random_episode(), experiment.ag.eval_episode()
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
@click.pass_context
def random(ctx: click.Context, save: bool, render: bool, replay: bool) -> None:
    experiment = ctx.obj["experiment"]
    if save:
        experiment.random(render=render, replay=replay, action_file=action_file)
    else:
        experiment.random(render=render, replay=replay)


@rainy_cli.command(help="Given a save file and restart training")
@click.pass_context
@click.argument("logdir", type=str)
@click.option(
    "--additional-steps",
    type=int,
    default=100,
    help="The number of  additional training steps",
)
@click.option(
    "--eval-render", is_flag=True, help="Render the environment when evaluating"
)
def retrain(
    ctx: click.Context, logdir: str, additional_steps: int, eval_render: bool
) -> None:
    experiment = ctx.obj["experiment"]
    experiment.retrain(logdir, additional_steps, eval_render)
    if mpi.IS_MPI_ROOT:
        print(
            "random play: {}, trained: {}".format(
                experiment.ag.random_episode(), experiment.ag.eval_episode()
            )
        )


@rainy_cli.command(help="Load a specified save file and evaluate the agent")
@click.argument("logdir", type=str)
@click.option("--render", is_flag=True, help="Render the agent")
@click.option(
    "--replay",
    is_flag=True,
    help="Show replay(works only with special environments, e.g., rogue-gym)",
)
@click.pass_context
def eval(ctx: click.Context, logdir: str, render: bool, replay: bool,) -> None:
    experiment = ctx.obj["experiment"]
    experiment.evaluate(logdir, render=render, replay=replay)


def _add_options(options: List[click.Command]) -> click.Group:
    for option in options:
        rainy_cli.params.append(option)


def run_cli(
    config_gen: Callable[..., Config],
    agent_gen: Callable[[Config], Agent],
    script_path: Optional[str] = None,
    options: List[click.Command] = [],
) -> None:
    obj = {
        "config_gen": config_gen,
        "agent_gen": agent_gen,
        "script_path": script_path,
    }
    _add_options(options)
    rainy_cli(obj=obj)
