import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from .agents import Agent, EpisodeResult
from .prelude import Array
from .utils.log import ExperimentLog
from .utils.misc import has_freq_in_interval
from .lib.mpi import IS_MPI_ROOT

SAVE_FILE_DEFAULT = 'rainy-agent.pth'
SAVE_FILE_OLD = 'rainy-agent.save'
ACTION_FILE_DEFAULT = 'actions.json'


def eval_impl(
        ag: Agent,
        save_file: Optional[Path],
        render: bool = False,
        replay: bool = False,
) -> List[EpisodeResult]:
    n = ag.config.eval_times
    ag.set_mode(train=False)
    if save_file is not None and save_file.is_file():
        res = [ag.eval_and_save(save_file.as_posix(), render=render) for _ in range(n)]
    elif hasattr(ag, 'eval_parallel') and not render and not replay:
        res = ag.eval_parallel(n)  # type: ignore
    else:
        res = [ag.eval_episode(render=render) for _ in range(n)]
    ag.set_mode(train=True)
    return res


def _reward_and_length(results: List[EpisodeResult]) -> Tuple[Array[float], Array[float]]:
    rewards = np.array([t.reward for t in results])
    length = np.array([t.length for t in results])
    return rewards, length


def train_agent(
        ag: Agent,
        episode_offset: int = 0,
        saveid_start: int = 0,
        save_file_name: str = SAVE_FILE_DEFAULT,
        action_file_name: str = ACTION_FILE_DEFAULT,
) -> None:
    action_file = Path(action_file_name)

    def log_episode(episodes: int, res: List[EpisodeResult]) -> None:
        rewards, length = _reward_and_length(res)
        ag.logger.exp('train', {
            'episodes': episodes + episode_offset,
            'update-steps': ag.update_steps,
            'reward-mean': float(np.mean(rewards)),
            'reward-min': float(np.min(rewards)),
            'reward-max': float(np.max(rewards)),
            'reward-stdev': float(np.std(rewards)),
            'length-mean': int(np.mean(length)),
        })

    def log_eval() -> None:
        log_dir = ag.logger.log_dir
        if ag.config.save_eval_actions and log_dir:
            fname = log_dir.joinpath('{}-{}{}'.format(
                action_file.stem,
                episodes,
                action_file.suffix
            ))
            res = eval_impl(ag, fname)
        else:
            res = eval_impl(ag, None)
        rewards, length = _reward_and_length(res)
        ag.logger.exp('eval', {
            'total-steps': ag.total_steps,
            'update-steps': ag.update_steps,
            'reward-mean': float(np.mean(rewards)),
            'reward-min': float(np.min(rewards)),
            'reward-max': float(np.max(rewards)),
            'reward-stdev': float(np.std(rewards)),
            'length-mean': float(np.mean(length)),
        })

    def truncate_episode(episodes: int, freq: Optional[int]) -> int:
        return episodes - episodes % freq if freq else episodes

    episodes = 0
    steps = 0
    save_id = saveid_start
    results: List[EpisodeResult] = []

    for res in ag.train_episodes(ag.config.max_steps):
        ep_len = len(res)
        episodes += ep_len
        step_diff = ag.total_steps - steps
        steps = ag.total_steps
        results += res
        if has_freq_in_interval(episodes, ep_len, ag.config.episode_log_freq) and IS_MPI_ROOT:
            eps = truncate_episode(episodes, ag.config.episode_log_freq)
            log_episode(eps, results[:eps])
            results = results[eps:]
        if has_freq_in_interval(steps, step_diff, ag.config.eval_freq) and IS_MPI_ROOT:
            log_eval()
        if has_freq_in_interval(steps, step_diff, ag.config.save_freq) and IS_MPI_ROOT:
            ag.save(save_file_name + '.{}'.format(save_id))
            save_id += 1
    log_eval()
    ag.save(save_file_name)
    ag.close()


def _load_agent(load_file_name: str, logdir_path: Path, ag: Agent) -> bool:
    def _try_load(fname: str) -> bool:
        p = logdir_path.joinpath(fname)
        if p.exists():
            ag.load(p.as_posix())
            return True
        else:
            return False
    while _try_load(load_file_name) is False:
        if load_file_name == SAVE_FILE_DEFAULT:
            load_file_name = SAVE_FILE_OLD
            continue
        return False
    return True


def retrain_agent(
        ag: Agent,
        log: ExperimentLog,
        load_file_name: str = SAVE_FILE_DEFAULT,
        additional_steps: int = 100
) -> None:
    path = log.log_path.parent
    if not _load_agent(load_file_name, path, ag):
        raise ValueError('Load file {} does not exists'.format(load_file_name))
    episodes, total = 0, 0
    for d in reversed(log.unwrapped):
        if episodes == 0 and 'episodes' in d:
            episodes = d['episodes']
        if total == 0 and 'total-steps' in d:
            total = d['total-steps']
        if episodes > 0 and total > 0:
            break
    save_files = [f for f in path.glob(SAVE_FILE_DEFAULT + '.*')]
    if len(save_files) > 0:
        save_id = len(save_files)
    else:
        save_id = 0
    ag.total_steps = total
    ag.config.max_steps += additional_steps
    train_agent(ag, episode_offset=episodes, saveid_start=save_id)


def eval_agent(
        ag: Agent,
        log_dir: str,
        load_file_name: str = SAVE_FILE_DEFAULT,
        render: bool = False,
        replay: bool = False,
        action_file: Optional[str] = None,
) -> None:
    path = Path(log_dir)
    if not _load_agent(load_file_name, path, ag):
        raise ValueError('Load file {} does not exists'.format())
    save_file = path.joinpath if ag.config.save_eval_actions and action_file else None
    res = eval_impl(ag, save_file, render, replay)
    print('{}'.format(res))
    if render:
        input('--Press Enter to exit--')
    if replay:
        try:
            ag.config.eval_env.unwrapped.replay()
        except Exception:
            print('--replay was specified, but environment has no function named replay')
    ag.close()


def random_agent(
        ag: Agent,
        render: bool = False,
        replay: bool = False,
        action_file: Optional[str] = None
) -> None:
    if action_file:
        res = ag.random_and_save(action_file, render=render)
    else:
        res = ag.random_episode(render=render)
    print('{}'.format(res))
    if replay:
        try:
            ag.config.eval_env.unwrapped.replay()
        except Exception:
            print('--replay was specified, but environment has no function named replay')
    ag.close()
