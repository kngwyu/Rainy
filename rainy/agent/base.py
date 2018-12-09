from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
from pathlib import Path
import torch
from torch import nn
from typing import Callable, Iterable, List, Tuple
from ..config import Config
from ..envs import Action, EnvExt, State


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
    def best_action(self, state: State) -> Action:
        """Return the best action according to training results.
        """
        pass

    def num_envs(self) -> int:
        return 1

    def random_action(self) -> Action:
        return np.random.randint(self.env.action_dim)

    def __eval_episode(self, select_action: Callable[[State], Action]) -> Tuple[float, EnvExt]:
        total_reward = 0.0
        steps = 0
        env = self.config.eval_env
        if self.config.seed:
            env.seed(self.config.seed)
        state = env.reset()
        while True:
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
        return self.__eval_episode(act)[0]

    def random_and_save(self, fname: str) -> float:
        def act(_state) -> Action:
            return self.random_action()
        res, env = self.__eval_episode(act)
        env.save_history(fname)
        return res

    def eval_episode(self) -> float:
        return self.__eval_episode(self.best_action)[0]

    def eval_and_save(self, fname: str) -> float:
        res, env = self.__eval_episode(self.best_action)
        env.save_history(fname)
        return res

    def save(self, filename: str) -> None:
        save_dict = {}
        for idx, member_str in enumerate(self.members_to_save()):
            value = getattr(self, member_str)
            if isinstance(value, nn.Module):
                save_dict[idx] = value.state_dict()
            else:
                save_dict[idx] = value
        log_dir = self.config.logger.log_dir
        if log_dir is None:
            log_dir = Path('.')
        torch.save(save_dict, log_dir.joinpath(filename))

    def load(self, filename: str) -> None:
        saved_dict = torch.load(filename)
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
    @abstractmethod
    def nstep(self, states: List[State], nstep: int) -> Tuple[List[State], List[float]]:
        pass

    def step_len(self) -> int:
        return self.config.nstep

    def num_envs(self) -> int:
        return

    def train_episode(self) -> List[float]:
        total_reward = 0.0
        steps = 0
        if self.config.seed:
            self.penv.seed(self.config.seed)
        states = self.env.reset()
        nstep = self.step_len()
        while True:
            states, rewards = self.nstep(states, nstep)
            steps += nstep
            self.total_steps += nstep
            if rewards:
                return rewards
        return []
