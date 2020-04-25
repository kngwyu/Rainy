import os

import rainy
from rainy.agents import DQNAgent
from rainy.envs import ClassicControl, MultiProcEnv


@rainy.main(agent=DQNAgent, script_path=os.path.realpath(__file__))
def main(
    envname: str = "CartPole-v0", max_steps: int = 100000, nworkers: int = 1
) -> rainy.Config:
    c = rainy.Config()
    c.set_env(lambda: ClassicControl(envname))
    c.set_parallel_env(MultiProcEnv)
    c.max_steps = max_steps
    c.episode_log_freq = 100
    c.nworkers = nworkers
    c.replay_batch_size = 64 * nworkers
    return c


if __name__ == "__main__":
    main()
