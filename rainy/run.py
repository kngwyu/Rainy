from .agent import Agent


def __interval(turn: int, freq: int) -> bool:
    return turn != 0 and turn % freq == 0


def train_agent(ag: Agent, save_file_name: str = 'rainy-agent.save') -> None:
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

