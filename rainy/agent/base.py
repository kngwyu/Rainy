from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
import torch
from torch import nn
from typing import Any, Optional, Tuple
from ..config import Config
from ..env_ext import Action, EnvExt


class Agent(ABC):
    """Children must call super().__init__(config) first
    """
    def __init__(self, config: Config) -> None:
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
    def best_action(self, state: ndarray) -> Action:
        pass

    @abstractmethod
    def step(self, state: ndarray) -> Tuple[ndarray, float, bool]:
        pass

    def eval_episode(self, eval_env: Optional[EnvExt] = None) -> float:
        total_reward = 0.0
        steps = 0
        if eval_env is None:
            env = self.env
            env.seed(self.config.seed)
        else:
            env = eval_env
        state = env.reset()
        while True:
            action = self.best_action(state)
            state, reward, done, _ = env.step(action)
            steps += 1
            total_reward += reward
            if done:
                break
        return total_reward

    def episode(self) -> float:
        total_reward = 0.0
        steps = 0
        self.env.seed(self.config.seed)
        state = self.env.reset()
        while True:
            state, reward, done = self.step(state)
            steps += 1
            self.total_steps += 1
            total_reward += reward
            if done:
                break
        return total_reward

    def save(self, filename: str) -> None:
        save_dict = {}
        for idx, member_str in enumerate(self.members_to_save()):
            value = getattr(self, member_str)
            if isinstance(value, nn.Module):
                save_dict[idx] = value.state_dict()
            else:
                save_dict[idx] = value
        torch.save(save_dict, filename)

    def load(self, filename: str) -> None:
        saved_dict = torch.load(filename)
        for idx, member_str in enumerate(self.members_to_save()):
            saved_item = saved_dict[idx]
            mem = getattr(self, member_str)
            if isinstance(mem, nn.Module):
                mem.load_state_dict(saved_item)
            else:
                setattr(self, member_str, saved_item)
