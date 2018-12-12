import numpy as np
from numpy import ndarray
import torch
from torch import nn
from typing import Tuple
from .base import NStepAgent
from ..config import Config
from ..envs import Action, State


class NStepDqnAgent(NStepAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        self.net = config.net('value')
        self.target_net = config.net('value')
        self.optimizer = config.optimizer(self.net.parameters())
        self.target_net.load_state_dict(self.net.state_dict())

        self.criterion = nn.MSELoss()
        self.policy = config.explorer(self.net)

    def members_to_save(self) -> Tuple[str, ...]:
        return "net", "target_net", "policy", "total_steps"

    def nstep(self, state: State) -> Tuple[ndarray, float, bool]:
        pass
