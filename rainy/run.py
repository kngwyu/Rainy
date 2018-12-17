from .agent import Agent
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
    rewards: List[float] = []
    action_file = Path(action_file_name)

    def log_episode(episodes: int, rewards: np.ndarray) -> None:
        ag.logger.exp('train_reward', {
            'episodes': episodes,
            'update_steps': ag.update_steps,
            'reward_mean': float(np.mean(rewards)),
            'reward_max': float(np.max(rewards)),
            'reward_min': float(np.min(rewards)),
        })

    def log_eval(episodes: int):
        log_dir = ag.logger.log_dir
        if ag.config.save_eval_actions and log_dir:
            fname = log_dir.joinpath('{}-{}{}'.format(
                action_file.stem,
                episodes,
                action_file.suffix
            ))
            reward = ag.eval_and_save(fname.as_posix())
        else:
            reward = ag.eval_episode()
            ag.logger.exp('eval', {
                'episodes': episodes,
                'total_steps': ag.total_steps,
                'reward': reward,
            })

    def interval(turn: int, width: int, freq: Optional[int]) -> bool:
        return freq and turn != 0 and turn // freq != (turn - width) // freq  # type: ignore

    def truncate_episode(episodes: int, freq: Optional[int]) -> int:
        return episodes - episodes % freq if freq else episodes

    for rw in ag.train_episodes(max_steps):
        rw_len = len(rw)
        episodes += rw_len
        rewards += rw
        if interval(episodes, rw_len, ag.config.episode_log_freq):
            eps = truncate_episode(episodes, ag.config.episode_log_freq)
            log_episode(eps, np.array(rewards[:eps]))
            rewards = rewards[eps:]
        if interval(episodes, rw_len, ag.config.eval_freq):
            log_eval(truncate_episode(episodes, ag.config.eval_freq))
        if interval(episodes, rw_len, ag.config.save_freq):
            ag.save(save_file_name)
    log_eval(episodes)
    ag.save(save_file_name)
    ag.close()


def eval_agent(
        ag: Agent,
        log_dir: str,
        load_file_name: str = SAVE_FILE_DEFAULT,
        render: bool = False,
        action_file: Optional[str] = None
) -> None:
    path = Path(log_dir)
    ag.load(path.joinpath(load_file_name).as_posix())
    if action_file:
        res = ag.eval_and_save(path.joinpath(action_file).as_posix(), render=render)
    else:
        res = ag.eval_episode(render=render)
    print('reward: {}'.format(res))
    if render:
        input('--Press Enter to exit--')
    ag.close()

