from abc import ABC, abstractmethod
import numpy as np
from pathlib import Path
import torch
from torch import nn
from typing import Callable, List, Tuple
from ..config import Config
from .nstep_common import RolloutStorage
from ..envs import Action, EnvExt, State
from ..util.meta import Array


class Agent(ABC):
    """Children must call super().__init__(config) first
    """
    def __init__(self, config: Config) -> None:
        self.logger = config.logger
        self.config = config
        self.env = config.env()
        self.total_steps = 0

    @abstractmethod
    def members_to_save(self) -> Tuple[str, ...]:
        """Here you can specify members you want to save.

    Examples::
        def members_to_save(self):
            return "net", "target"
        """
        pass

    @abstractmethod
    def train_episode(self) -> List[float]:
        """Train the agent.
        """
        pass

    @abstractmethod
    def eval_action(self, state: State) -> Action:
        """Return the best action according to training results.
        """
        pass

    def close(self) -> None:
        self.env.close()

    def num_envs(self) -> int:
        return 1

    def random_action(self) -> Action:
        return np.random.randint(self.env.action_dim)

    def __eval_episode(
            self,
            select_action: Callable[[State], Action],
            render: bool
    ) -> Tuple[float, EnvExt]:
        total_reward = 0.0
        steps = 0
        env = self.config.eval_env
        if self.config.seed:
            env.seed(self.config.seed)
        state = env.reset()
        while True:
            if render:
                env.render()
            state = env.state_to_array(state)
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            steps += 1
            total_reward += reward
            if done:
                break
        return total_reward, env

    def random_episode(self) -> float:
        def act(_state) -> Action:
            return self.random_action()
        return self.__eval_episode(act, False)[0]

    def random_and_save(self, fname: str) -> float:
        def act(_state) -> Action:
            return self.random_action()
        res, env = self.__eval_episode(act, False)
        env.save_history(fname)
        return res

    def eval_episode(self, render: bool = False) -> float:
        return self.__eval_episode(self.eval_action, render=render)[0]

    def eval_and_save(self, fname: str, render: bool = False) -> float:
        res, env = self.__eval_episode(self.eval_action, render=render)
        env.save_history(fname)
        return res

    def save(self, filename: str) -> None:
        save_dict = {}
        for idx, member_str in enumerate(self.members_to_save()):
            value = getattr(self, member_str)
            if isinstance(value, nn.DataParallel):
                save_dict[idx] = value.module.state_dict()
            elif isinstance(value, nn.Module):
                save_dict[idx] = value.state_dict()
            else:
                save_dict[idx] = value
        log_dir = self.config.logger.log_dir
        if log_dir is None:
            log_dir = Path('.')
        torch.save(save_dict, log_dir.joinpath(filename))

    def load(self, filename: str) -> None:
        saved_dict = torch.load(filename, map_location=self.config.device.unwrapped)
        for idx, member_str in enumerate(self.members_to_save()):
            saved_item = saved_dict[idx]
            mem = getattr(self, member_str)
            if isinstance(mem, nn.Module):
                mem.load_state_dict(saved_item)
            else:
                setattr(self, member_str, saved_item)


class OneStepAgent(Agent):
    @abstractmethod
    def step(self, state: State) -> Tuple[State, float, bool]:
        pass

    def train_episode(self) -> List[float]:
        total_reward = 0.0
        steps = 0
        if self.config.seed:
            self.env.seed(self.config.seed)
        state = self.env.reset()
        while True:
            state, reward, done = self.step(state)
            steps += 1
            self.total_steps += 1
            total_reward += reward
            if done:
                break
        return [total_reward]


class NStepAgent(Agent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.storage = RolloutStorage(config.nsteps, config.nworkers, config.device)
        self.rewards = np.zeros(config.nworkers, dtype=np.float32)
        self.penv = config.parallel_env()

    @abstractmethod
    def nstep(self, states: Array[State]) -> Tuple[Array[State], Array[float]]:
        pass

    def close(self) -> None:
        self.env.close()
        self.penv.close()

    def train_episode(self) -> List[float]:
        if self.storage.initialized():
            states = self.storage.states[0]
        else:
            if self.config.seed:
                self.penv.seed(self.config.seed)
            states = self.penv.reset()
            self.storage.set_initial_state(states)
        step = self.config.nsteps * self.config.nworkers
        while True:
            states, rewards = self.nstep(states)
            self.total_steps += step
            if rewards:
                break
        return rewards
