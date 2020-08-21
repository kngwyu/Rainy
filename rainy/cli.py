from typing import Callable, Optional, Tuple, Type, Union

import click

from .agents import Agent
from .config import Config
from .experiment import Experiment
from .lib import mpi


@click.group()
@click.option("--model", type=str, default=None, help="Name of the save file")
@click.option(
    "--action-file", type=str, default="actions.json", help="Name of the action file",
)
@click.pass_context
def rainy_cli(
    ctx: click.Context, model: Optional[str], action_file: Optional[str], **kwargs,
) -> None:
    config = ctx.obj.config_gen(**kwargs)
    ag = ctx.obj.get_agent(config, **kwargs)
    ctx.obj.experiment = Experiment(ag, model, action_file)


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
            comment="" if comment is None else comment, kwargs=ctx.obj.kwargs,
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
@click.argument("logdir-or-file", type=str)
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
    ctx: click.Context, logdir_or_file: str, additional_steps: int, eval_render: bool,
) -> None:
    experiment = ctx.obj.experiment
    experiment.retrain(logdir_or_file, additional_steps, eval_render)
    if mpi.IS_MPI_ROOT:
        print(
            "random play: {}, trained: {}".format(
                experiment.ag.random_episode(), experiment.ag.eval_episode()
            )
        )


@rainy_cli.command(help="Load a specified save file and evaluate the agent")
@click.argument("logdir-or-file", type=str)
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
    ctx: click.Context,
    logdir_or_file: str,
    save: bool,
    render: bool,
    pause: bool,
    replay: bool,
) -> None:
    ctx.obj.experiment.config.save_eval_actions |= save
    ctx.obj.experiment.load_and_evaluate(logdir_or_file, render, replay, pause)


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


def subcommand(*args, **kwargs) -> callable:
    def decorator(f: callable) -> callable:
        return rainy_cli.command(*args, **kwargs)(click.pass_context(f))

    return decorator


def option(*param_decls, **attrs) -> callable:
    def decorator(f):
        option_attrs = attrs.copy()
        if "help" in option_attrs:
            import inspect

            option_attrs["help"] = inspect.cleandoc(option_attrs["help"])
        option = click.Option(param_decls, **attrs)
        rainy_cli.params.append(option)
        return f

    return decorator


def _is_optional(cls: type) -> bool:
    if not hasattr(cls, "__origin__") or cls.__origin__ is not Union:
        return False

    if len(cls.__args__) != 2 or cls.__args__[1]() is not None:
        raise TypeError(f"Invalid type for CLI arguments: {cls}")
    return True


def _defaults(f: callable) -> Tuple[str]:
    defaults = f.__defaults__
    if defaults is None:
        defaults = ()
    if f.__kwdefaults__ is None:
        return defaults
    else:
        return defaults + tuple(f.__kwdefaults__.values())


def _annot_to_clargs(f: callable) -> None:
    annot = f.__annotations__
    # Ignore return type
    if "return" in annot:
        del annot["return"]
    defaults = _defaults(f)
    has_default_min = len(annot) - len(defaults)
    used_names = [param.name for param in rainy_cli.params]
    for i, name in enumerate(annot.keys()):
        if name in used_names:
            continue
        cls = annot[name]
        if has_default_min <= i:
            default = {"default": defaults[i - has_default_min]}
            value = defaults[i - has_default_min]
        else:
            default = {}
            value = None
        if cls is bool and value is False:
            option = click.Option(["--" + name.replace("_", "-")], is_flag=True,)
        elif _is_optional(cls):
            option = click.Option(
                ["--" + name.replace("_", "-")], type=cls.__args__[0], **default,
            )
        else:
            option = click.Option(
                ["--" + name.replace("_", "-")], type=annot[name], **default,
            )
        rainy_cli.params.append(option)


def main(
    agent: Optional[Type[Agent]] = None,
    script_path: Optional[str] = None,
    agent_selector: Optional[Callable[..., Type[Agent]]] = None,
):
    def decorator(f):
        if hasattr(f, "__annotations__"):
            _annot_to_clargs(f)

        rainy_cli(obj=_CLIContext(f, agent, agent_selector, script_path))
        return f

    return decorator
