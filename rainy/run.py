import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from .agents import Agent, EpisodeResult
from .prelude import Array
from .utils.misc import has_freq_in_interval
from .lib.mpi import IS_MPI_ROOT

SAVE_FILE_DEFAULT = "rainy-agent.pth"
SAVE_FILE_OLD = "rainy-agent.save"
ACTION_FILE_DEFAULT = "actions.json"


def train_agent(
    ag: Agent,
    episode_offset: int = 0,
    saveid_start: int = 0,
    save_file_name: str = SAVE_FILE_DEFAULT,
    action_file_name: str = ACTION_FILE_DEFAULT,
) -> None:
    ag.logger.summary_setting(
        "train",
        ["episodes", "total_steps", "update_steps"],
        interval=ag.config.episode_log_freq,
        color="red",
    )
    ag.logger.summary_setting(
        "eval", ["total_steps", "update_steps"], color="green", dtype_is_array=True,
    )
    action_file = Path(action_file_name)

    def log_episode(epsodes: int, results: List[EpisodeResult]) -> None:
        for i, res in enumerate(results):
            ag.logger.submit(
                "train",
                episodes=episodes + i + episode_offset,
                total_steps=ag.total_steps,
                update_steps=ag.update_steps,
                reward=res.reward,
                length=res.length,
            )

    def log_eval() -> None:
        logdir = ag.logger.logdir
        if ag.config.save_eval_actions and logdir:
            fname = logdir.joinpath(
                "{}-{}{}".format(action_file.stem, episodes, action_file.suffix)
            )
            res = _eval_impl(ag, fname)
        else:
            res = _eval_impl(ag, None)
        rewards, length = _reward_and_length(res)
        ag.logger.submit(
            "eval",
            total_steps=ag.total_steps,
            update_steps=ag.update_steps,
            rewards=rewards,
            length=length,
        )

    episodes = 0
    steps = 0
    save_id = saveid_start

    for res in ag.train_episodes(ag.config.max_steps):
        ep_len = len(res)
        if ep_len > 0 and IS_MPI_ROOT:
            log_episode(episodes, res)
        episodes += ep_len
        step_diff = ag.total_steps - steps
        steps = ag.total_steps
        if has_freq_in_interval(steps, step_diff, ag.config.eval_freq) and IS_MPI_ROOT:
            log_eval()
        if has_freq_in_interval(steps, step_diff, ag.config.save_freq) and IS_MPI_ROOT:
            ag.save(save_file_name + ".{}".format(save_id))
            save_id += 1
    log_eval()
    ag.save(save_file_name)
    ag.close()


def _eval_impl(
    ag: Agent, save_file: Optional[Path], render: bool = False, replay: bool = False,
) -> List[EpisodeResult]:
    n = ag.config.eval_times
    ag.set_mode(train=False)
    if save_file is not None and save_file.is_file():
        res = [ag.eval_and_save(save_file.as_posix(), render=render) for _ in range(n)]
    elif hasattr(ag, "eval_parallel") and not render and not replay:
        res = ag.eval_parallel(n)  # type: ignore
    else:
        res = [ag.eval_episode(render=render) for _ in range(n)]
    ag.set_mode(train=True)
    return res


def _reward_and_length(
    results: List[EpisodeResult],
) -> Tuple[Array[float], Array[float]]:
    rewards = np.array([t.reward for t in results])
    length = np.array([t.length for t in results])
    return rewards, length


def _load_agent(file_name: str, logdir: Path, ag: Agent) -> bool:
    p = logdir.joinpath(file_name)
    if p.exists():
        ag.load(p.as_posix())
        return True
    return False


def retrain_agent(
    ag: Agent,
    logdir_: str,
    load_file_name: str = SAVE_FILE_DEFAULT,
    additional_steps: int = 100,
) -> None:
    logdir = Path(logdir_)
    if not _load_agent(load_file_name, logdir, ag):
        raise ValueError("Load file {} does not exists".format(load_file_name))
    total_steps, episodes = ag.logger.retrive(logdir)
    save_files = list(logdir.glob(SAVE_FILE_DEFAULT + ".*"))
    if len(save_files) > 0:
        save_id = len(save_files)
    else:
        save_id = 0
    ag.total_steps = total_steps
    ag.config.max_steps += additional_steps
    train_agent(ag, episode_offset=episodes, saveid_start=save_id)


def eval_agent(
    ag: Agent,
    logdir: str,
    load_file_name: str = SAVE_FILE_DEFAULT,
    render: bool = False,
    replay: bool = False,
    action_file: Optional[str] = None,
) -> None:
    path = Path(logdir)
    if not _load_agent(load_file_name, path, ag):
        raise ValueError("Load file {} does not exists".format())
    save_file = path.joinpath if ag.config.save_eval_actions and action_file else None
    res = _eval_impl(ag, save_file, render, replay)
    print("{}".format(res))
    if render:
        input("--Press Enter to exit--")
    if replay:
        try:
            ag.config.eval_env.unwrapped.replay()
        except Exception:
            print(
                "--replay was specified, but environment has no function named replay"
            )
    ag.close()


def random_agent(
    ag: Agent,
    render: bool = False,
    replay: bool = False,
    action_file: Optional[str] = None,
) -> None:
    if action_file:
        res = ag.random_and_save(action_file, render=render)
    else:
        res = ag.random_episode(render=render)
    print("{}".format(res))
    if replay:
        try:
            ag.config.eval_env.unwrapped.replay()
        except Exception:
            print(
                "--replay was specified, but environment has no function named replay"
            )
    ag.close()
