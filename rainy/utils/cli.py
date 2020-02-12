from typing import Callable, List, Optional, Type

import click

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
    if envname is not None:
        kwargs["envname"] = envname
    if max_steps is not None:
        kwargs["max_steps"] = max_steps
    config = ctx.obj.config_gen(**kwargs)
    config.seed = seed
    ag = ctx.obj.get_agent(config, **kwargs)
    ctx.obj.experiment = Experiment(ag, model, action_file)
    if envname is not None:
        ctx.obj.envname = envname


@rainy_cli.command(help="Train agents")
@click.pass_context
@click.option(
    "--comment",
    type=str,
    default=None,
    help="Comment that would be wrote to fingerprint.txt",
)
@click.option("--logdir", type=str, default=None, help="Name of the log directory")
@click.option(
    "--eval-render", is_flag=True, help="Render the environment when evaluating"
)
def train(
    ctx: click.Context,
    comment: Optional[str],
    logdir: Optional[str],
    eval_render: bool = True,
) -> None:
    experiment = ctx.obj.experiment
    script_path = ctx.obj.script_path
    if script_path is not None:
        fingerprint = dict(
            comment="" if comment is None else comment,
            envname=ctx.obj.envname,
            kwargs=ctx.obj.kwargs,
        )
        experiment.logger.setup_from_script_path(
            script_path, dirname=logdir, fingerprint=fingerprint
        )
    experiment.train(eval_render=eval_render)
    if mpi.IS_MPI_ROOT:
        print(
            "random play: {}, trained: {}".format(
                experiment.ag.random_episode(), experiment.ag.eval_episode()
            )
        )


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
    experiment = ctx.obj.experiment
    experiment.retrain(logdir, additional_steps, eval_render)
    if mpi.IS_MPI_ROOT:
        print(
            "random play: {}, trained: {}".format(
                experiment.ag.random_episode(), experiment.ag.eval_episode()
            )
        )


@rainy_cli.command(help="Load a specified save file and evaluate the agent")
@click.argument("logdir", type=str)
@click.option("--save", is_flag=True, help="Save actions")
@click.option("--render", is_flag=True, help="Render the agent")
@click.option("--pause", is_flag=True, help="Pause before replay. Used for screencast")
@click.option(
    "--replay",
    is_flag=True,
    help="Show replay(works only with special environments, e.g., rogue-gym)",
)
@click.pass_context
def eval(
    ctx: click.Context, logdir: str, save: bool, render: bool, pause: bool, replay: bool
) -> None:
    ctx.obj.experiment.config.save_eval_actions |= save
    ctx.obj.experiment.load_and_evaluate(logdir, render, replay, pause)


@rainy_cli.command(help="Run the random agent and show its result")
@click.option("--save", is_flag=True, help="Save actions")
@click.option("--render", is_flag=True, help="Render the agent")
@click.option("--pause", is_flag=True, help="Pause before replay. Used for screencast")
@click.option(
    "--replay",
    is_flag=True,
    help="Show replay(works only with special environments, e.g., rogue-gym)",
)
@click.pass_context
def random(
    ctx: click.Context, save: bool, render: bool, pause: bool, replay: bool
) -> None:
    ctx.obj.experiment.config.save_eval_actions |= save
    ctx.obj.experiment.random(render, replay, pause)


def _add_options(options: List[click.Command]) -> None:
    for option in options:
        rainy_cli.params.append(option)


class _CLIContext:
    def __init__(
        self,
        config_gen: Callable[..., Config],
        agent_cls: Type[Agent],
        agent_selector: Callable[..., Type[Agent]],
        script_path: Optional[str],
    ) -> None:
        self.config_gen = config_gen
        self.agent_cls = agent_cls
        self.agent_selector = agent_selector
        self.script_path = script_path
        self.experiment = None
        self.envname = "Default"
        self.kwargs = {}

    def get_agent(self, config: Config, **kwargs) -> Agent:
        self.kwargs.update(kwargs)
        if self.agent_cls is not None:
            return self.agent_cls(config)
        elif self.agent_selector is not None:
            agent_cls = self.agent_selector(**kwargs)
            return agent_cls(config)
        else:
            assert False, "Unreachable!"


def run_cli(
    config_gen: Callable[..., Config],
    agent_cls: Optional[Type[Agent]] = None,
    script_path: Optional[str] = None,
    options: List[click.Command] = [],
    agent_selector: Optional[Callable[..., Type[Agent]]] = None,
) -> None:
    if agent_cls is None and agent_selector is None:
        raise ValueError("run_cli needs agent_cls or agent_selector!")

    _add_options(options)
    rainy_cli(obj=_CLIContext(config_gen, agent_cls, agent_selector, script_path))
