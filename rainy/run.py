import numpy as np
from pathlib import Path
from typing import List, Optional, Tuple
from .agents import Agent, EpisodeResult
from .prelude import Array

SAVE_FILE_DEFAULT = 'rainy-agent.pth'
SAVE_FILE_OLD = 'rainy-agent.save'
ACTION_FILE_DEFAULT = 'actions.json'


def eval_fn(
        ag: Agent,
        save_file: Optional[Path],
        render: bool = False,
) -> List[EpisodeResult]:
    n = ag.config.eval_times
    ag.set_mode(train=False)
    if save_file is not None:
        res = [ag.eval_and_save(save_file.as_posix(), render=render) for _ in range(n)]
    else:
        res = [ag.eval_episode(render=render) for _ in range(n)]
    ag.set_mode(train=True)
    return res


def _reward_and_length(results: List[EpisodeResult]) -> Tuple[Array[float], Array[float]]:
    rewards = np.array(list(map(lambda t: t.reward, results)))
    length = np.array(list(map(lambda t: t.length, results)))
    return rewards, length


def train_agent(
        ag: Agent,
        save_file_name: str = SAVE_FILE_DEFAULT,
        action_file_name: str = ACTION_FILE_DEFAULT,
) -> None:
    action_file = Path(action_file_name)

    def log_episode(episodes: int, res: List[EpisodeResult]) -> None:
        rewards, length = _reward_and_length(res)
        ag.logger.exp('train', {
            'episodes': episodes,
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
            res = eval_fn(ag, fname, False)
        else:
            res = eval_fn(ag, None, False)
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

    def interval(turn: int, width: int, freq: Optional[int]) -> bool:
        return freq and turn != 0 and turn // freq != (turn - width) // freq  # type: ignore

    def truncate_episode(episodes: int, freq: Optional[int]) -> int:
        return episodes - episodes % freq if freq else episodes

    episodes = 0
    steps = 0
    results: List[EpisodeResult] = []
    save_id = 0

    for res in ag.train_episodes(ag.config.max_steps):
        ep_len = len(res)
        episodes += ep_len
        step_diff = ag.total_steps - steps
        steps = ag.total_steps
        results += res
        if interval(episodes, ep_len, ag.config.episode_log_freq):
            eps = truncate_episode(episodes, ag.config.episode_log_freq)
            log_episode(eps, results[:eps])
            results = results[eps:]
        if interval(steps, step_diff, ag.config.eval_freq):
            log_eval()
        if interval(steps, step_diff, ag.config.save_freq):
            ag.save(save_file_name + '.{}'.format(save_id))
            save_id += 1
    log_eval()
    ag.save(save_file_name)
    ag.close()


def eval_agent(
        ag: Agent,
        log_dir: str,
        load_file_name: str = SAVE_FILE_DEFAULT,
        render: bool = False,
        replay: bool = False,
        action_file: Optional[str] = None,
) -> None:
    path = Path(log_dir)

    def _try_load(fname: str) -> bool:
        p = path.joinpath(fname)
        if p.exists():
            ag.load(p.as_posix())
            return True
        else:
            return False
    while _try_load(load_file_name) is False:
        if load_file_name == SAVE_FILE_DEFAULT:
            load_file_name = SAVE_FILE_OLD
            continue
        raise ValueError('Load file {} does not exists'.format())
    if action_file is not None and len(action_file) > 0:
        res = eval_fn(ag, path.joinpath(action_file), render)
    else:
        res = eval_fn(ag, None, render)
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

