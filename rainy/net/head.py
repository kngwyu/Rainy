# network heads
from abc import ABC, abstractmethod
from numpy import ndarray
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Callable
from .body import NetworkBody
from .init import Initializer
from ..util import Device


class NetworkHead(nn.Module, ABC):
    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass


class LinearHead(NetworkHead):
    def __init__(self, input_dim: int, output_dim: int, init: Initializer = Initializer()) -> None:
        super(LinearHead, self).__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.fc = init(nn.Linear(input_dim, output_dim))

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)
