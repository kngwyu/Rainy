from .agent import Agent
import numpy as np
from pathlib import Path
from typing import Optional


SAVE_FILE_DEFAULT = 'rainy-agent.save'
ACTION_FILE_DEFAULT = 'actions.json'


def __interval(turn: int, width: int, freq: Optional[int]) -> bool:
    if freq:
        return turn != 0 and turn // freq != (turn - width) // freq
    else:
        return False


def train_agent(
        ag: Agent,
        save_file_name: str = SAVE_FILE_DEFAULT,
        action_file_name: str = ACTION_FILE_DEFAULT,
) -> None:
    max_steps = ag.config.max_steps
    episodes = 0
    rewards = []
    end = False
    action_file = Path(action_file_name)
    while not end:
        if max_steps and ag.total_steps > max_steps:
            end = True
        tmp = ag.train_episode()
        episodes += len(tmp)
        rewards += tmp
        if __interval(episodes, len(tmp),ag.config.episode_log_freq):
            rewards = np.array(rewards)
            ag.logger.exp('train_reward', {
                'episodes': episodes,
                'total_steps': ag.total_steps,
                'reward_mean': float(np.mean(rewards)),
                'reward_max': float(np.max(rewards)),
                'reward_min': float(np.min(rewards)),
            })
            rewards = []
        if end or __interval(episodes, len(tmp), ag.config.eval_freq):
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
        if end or __interval(episodes, len(tmp), ag.config.save_freq):
            ag.save(save_file_name)
    ag.close()


def eval_agent(
        ag: Agent,
        log_dir: str,
        load_file_name: str = SAVE_FILE_DEFAULT,
        action_file: Optional[str] = None
) -> None:
    path = Path(log_dir)
    ag.load(path.joinpath(load_file_name).as_posix())
    if action_file:
        res = ag.eval_and_save(path.joinpath(action_file).as_posix())
    else:
        res = ag.eval_episode()
    print('reward: {}'.format(res))

