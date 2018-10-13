import numpy as np
from numpy import ndarray
import torch
from torch import nn
from typing import Tuple
from .base import Agent
from ..config import Config
from ..envs import Action, State


class NstepDqnAgent(Agent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.net = config.value_net()
        self.target_net = config.value_net()
        self.optimizer = config.optimizer(self.net.parameters())
        self.target_net.load_state_dict(self.net.state_dict())

        self.criterion = nn.MSELoss()
        self.policy = config.explorer(self.net)
        self.replay = config.replay_buffer()
        self.batch_indices = torch.arange(
            config.batch_size,
            device=self.config.device(),
            dtype=torch.long
        )

    def members_to_save(self) -> Tuple[str, ...]:
        return "net", "target_net", "policy", "total_steps"

    def best_action(self, state: State) -> Action:
        action_values = self.net.action_values(state).detach()
        # Here supposes action_values is 1×(action_dim) array
        return action_values.argmax()

    def step(self, state: State) -> Tuple[ndarray, float, bool]:
        pass

    def sync_target_net(self) -> None:
        self.target_net.load_state_dict(self.net.state_dict())
