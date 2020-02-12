from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np

from ..prelude import Action, Array, State
from .base import Agent, DQNLikeAgent, EpisodeResult, Netout


class DQNLikeParallel(Agent):
    def __init__(self, agent: DQNLikeAgent) -> None:
        if not agent.SUPPORT_PARALLEL_ENV:
            raise RuntimeError(f"{(type(agent))} does not support parallel env!")

        super().__init__(agent.config)
        self.agent = agent
        self.penv = self.config.parallel_env()
        self.episode_length = np.zeros(self.config.nworkers, dtype=np.int)
        self.rewards = np.zeros(self.config.nworkers, dtype=np.float32)

    def _results(self, terminals: Array[bool], info: Array[dict]) -> None:
        res = []
        if self.penv.use_reward_monitor:
            for i in filter(lambda i: "episode" in i, info):
                res.append(EpisodeResult(i["episode"]["r"], i["episode"]["l"]))
        else:
            for i in filter(lambda i: terminals[i], range(self.config.nworkers)):
                res.append(EpisodeResult(self.rewards[i], self.episode_length[i]))
                self.rewards[i] = 0.0
                self.episode_length[i] = 0
        return res

    def step(self, states: Array[State]) -> tuple:
        actions = self.agent.batch_actions(states, self.penv)
        next_states, rewards, terminals, infos = self.penv.step(actions)
        for transition in zip(states, actions, next_states, rewards, terminals):
            self.agent.store_transition(*transition)
        return next_states, rewards, terminals, infos

    def train_episodes(self, max_steps: int) -> Iterable[List[EpisodeResult]]:
        self.config.set_parallel_seeds(self.penv)
        states = self.penv.reset()
        ag = self.agent
        while True:
            states, rewards, terminals, infos = self.step(states)
            # Train
            if ag.train_started and ag.total_steps % self.config.update_freq == 0:
                ag.train(ag.replay.sample(self.config.replay_batch_size))
                ag.update_steps += 1
            # Update stats
            self.rewards += rewards
            self.episode_length += 1
            ag.total_steps += self.config.nworkers
            self.total_steps = ag.total_steps
            results = self._results(terminals, infos)
            if len(results) > 0:
                yield results
            if self.total_steps >= max_steps:
                break

    def eval_action(self, state: Array, net_outputs: Optional[Netout] = None) -> Action:
        return self.agent.eval_action(state, net_outputs=net_outputs)

    @property
    def update_steps(self) -> int:
        return self.agent.update_steps

    def set_mode(self, train: bool = True) -> None:
        self.agent.set_mode(train)

    def close(self) -> None:
        self.agent.close()
        self.penv.close()

    def save(self, filename: str, directory: Optional[Path] = None) -> None:
        self.agent.save(filename, directory=directory)

    def load(self, filename: str, directory: Optional[Path] = None) -> bool:
        self.agent.load(filename, directory=directory)
