from .agent import Agent, EpisodeResult
import numpy as np
from pathlib import Path
from typing import List, Optional


SAVE_FILE_DEFAULT = 'rainy-agent.save'
ACTION_FILE_DEFAULT = 'actions.json'


def train_agent(
        ag: Agent,
        save_file_name: str = SAVE_FILE_DEFAULT,
        action_file_name: str = ACTION_FILE_DEFAULT,
) -> None:
    max_steps = ag.config.max_steps
    episodes = 0
    results: List[EpisodeResult] = []
    action_file = Path(action_file_name)

    def log_episode(episodes: int, res: List[EpisodeResult]) -> None:
        rewards = np.array(list(map(lambda t: t.reward, res)))
        length = np.array(list(map(lambda t: t.length, res)))
        ag.logger.exp('train', {
            'episodes': episodes,
            'update-steps': ag.update_steps,
            'reward-mean': float(np.mean(rewards)),
            'reward-min': float(np.min(rewards)),
            'reward-max': float(np.max(rewards)),
            'reward-stdev': float(np.std(rewards)),
            'length-mean': int(np.mean(length)),
        })

    def log_eval(episodes: int):
        log_dir = ag.logger.log_dir
        if ag.config.save_eval_actions and log_dir:
            fname = log_dir.joinpath('{}-{}{}'.format(
                action_file.stem,
                episodes,
                action_file.suffix
            ))
            res = ag.eval_and_save(fname.as_posix())
        else:
            res = ag.eval_episode()
        ag.logger.exp('eval', {
            'episodes': episodes,
            'update-steps': ag.update_steps,
            'reward': res.reward,
            'length': res.length,
        })

    def interval(turn: int, width: int, freq: Optional[int]) -> bool:
        return freq and turn != 0 and turn // freq != (turn - width) // freq  # type: ignore

    def truncate_episode(episodes: int, freq: Optional[int]) -> int:
        return episodes - episodes % freq if freq else episodes

    for res in ag.train_episodes(max_steps):
        ep_len = len(res)
        episodes += ep_len
        results += res
        if interval(episodes, ep_len, ag.config.episode_log_freq):
            eps = truncate_episode(episodes, ag.config.episode_log_freq)
            log_episode(eps, results[:eps])
            results = results[eps:]
        if interval(episodes, ep_len, ag.config.eval_freq):
            log_eval(truncate_episode(episodes, ag.config.eval_freq))
        if interval(episodes, ep_len, ag.config.save_freq):
            ag.save(save_file_name)
    log_eval(episodes)
    ag.save(save_file_name)
    ag.close()


def eval_agent(
        ag: Agent,
        log_dir: str,
        load_file_name: str = SAVE_FILE_DEFAULT,
        render: bool = False,
        replay: bool = False,
        action_file: Optional[str] = None
) -> None:
    path = Path(log_dir)
    ag.load(path.joinpath(load_file_name).as_posix())
    if action_file:
        res = ag.eval_and_save(path.joinpath(action_file).as_posix(), render=render)
    else:
        res = ag.eval_episode(render=render)
    print('{}'.format(res))
    if render:
        input('--Press Enter to exit--')
    if replay:
        try:
            ag.config.eval_env.unwrapped.replay()
        except Exception:
            print('--replay was specified, but environment has no function named replay')
    ag.close()

