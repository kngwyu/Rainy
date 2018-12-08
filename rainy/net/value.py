from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
from torch import nn, Tensor
from typing import Tuple, Union
from .body import DqnConv, FcBody, NetworkBody
from .head import LinearHead, NetworkHead
from ..util import Device


class ValuePredictor(ABC):
    @abstractmethod
    def action_values(self, state: ndarray, nostack: bool = False) -> Tensor:
        pass

    @property
    @abstractmethod
    def state_dims(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        pass


class ValueNet(ValuePredictor, nn.Module):
    """State -> [Value..]
    """
    def __init__(self, body: NetworkBody, head: NetworkHead, device: Device = Device()) -> None:
        assert body.output_dim == head.input_dim, \
            'body output and head input must have a same dimention'
        super(ValueNet, self).__init__()
        self.head = head
        if device.is_multi_gpu():
            self.body = device.data_parallel(body)
        else:
            self.body = body
        self.device = device
        self.to(self.device())

    def action_values(self, state: ndarray, nostack: bool = False) -> Tensor:
        if nostack:
            return self.forward(state)
        else:
            return self.forward(np.stack([state]))

    def forward(self, x: Union[ndarray, Tensor]) -> Tensor:
        x = self.device.tensor(x)
        x = self.body(x)
        x = self.head(x)
        return x

    @property
    def state_dims(self) -> Tuple[int, ...]:
        return self.body.input_dim

    @property
    def action_dim(self) -> int:
        return self.head.output_dim


def dqn_conv(state_dim: Tuple[int, ...], action_dim: int, device: Device) -> ValueNet:
    body = DqnConv(state_dim[0])
    head = LinearHead(body.output_dim, action_dim)
    return ValueNet(body, head, device=device)


def fc(state_dim: Tuple[int, ...], action_dim: int, device: Device) -> ValueNet:
    body = FcBody(state_dim[0])
    head = LinearHead(body.output_dim, action_dim)
    return ValueNet(body, head, device=device)

