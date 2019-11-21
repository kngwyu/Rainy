import gym
from gym.core import Wrapper
import time
from typing import Any, List, Tuple


class RewardMonitor(Wrapper):
    """Based on https://github.com/openai/baselines/blob/master/baselines/bench/monitor.py,
    but modiifed only to report raw rewards before wrapped by 'wrap_deepmind' or else.
    """

    def __init__(self, env: gym.Env) -> None:
        Wrapper.__init__(self, env=env)
        self.tstart = time.time()
        self.rewards: List[float] = []
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.episode_times: List[float] = []
        self.total_steps = 0

    def reset(self, **kwargs) -> Any:
        self.rewards = []
        return self.env.reset(**kwargs)

    def step(self, action) -> Tuple[Any, float, bool, dict]:
        ob, rew, done, info = self.env.step(action)
        self.update(ob, rew, done, info)
        return (ob, rew, done, info)

    def update(self, ob, rew, done, info) -> None:
        self.rewards.append(rew)
        self.total_steps += 1
        if not done:
            return
        if not isinstance(info, dict):
            raise ValueError("RewardMonitor assumes info is an instance of dict")
        eprew = sum(self.rewards)
        eplen = len(self.rewards)
        epinfo = {
            "r": round(eprew, 6),
            "l": eplen,
            "t": round(time.time() - self.tstart, 6),
        }
        self.episode_rewards.append(eprew)
        self.episode_lengths.append(eplen)
        self.episode_times.append(time.time() - self.tstart)
        info["episode"] = epinfo
