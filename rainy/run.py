from .agent import Agent
from .config import Config
from pathlib import Path
from typing import Optional


SAVE_FILE_DEFAULT = 'rainy-agent.save'


def __interval(turn: int, freq: Optional[int]) -> bool:
    if freq:
        return turn != 0 and turn % freq == 0
    else:
        return False


def train_agent(ag: Agent, save_file_name: str = SAVE_FILE_DEFAULT) -> None:
    max_steps = ag.config.max_steps
    episodes = 0
    rewards_sum = 0.0
    end = False
    while not end:
        if max_steps and ag.total_steps > max_steps:
            end = True
        rewards_sum += ag.episode()
        episodes += 1
        if __interval(episodes, ag.config.episode_log_freq):
            ag.logger.exp(
                'episodes: {}, total_steps: {}, rewards: {}'.format(
                    episodes,
                    ag.total_steps,
                    rewards_sum
                ))
            rewards_sum = 0
        if end or __interval(episodes, ag.config.eval_freq):
            ag.logger.exp('eval: {}'.format(ag.eval_episode()))
        if end or __interval(episodes, ag.config.save_freq):
            ag.save(save_file_name)


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
