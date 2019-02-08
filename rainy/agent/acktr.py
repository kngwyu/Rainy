import numpy as np
from torch import nn, Tensor
from torch.optim import Optimizer
from typing import Iterable, Union
from .a2c import A2cAgent
from ..config import Config
from ..envs import Action, State
from ..util.typehack import Array


class AcktrAgent(A2cAgent):
    """STUB
    """
    def nstep(self, states: Array[State]) -> Array[State]:
        pass

