from abc import ABC, abstractmethod
import numpy as np
from numpy import ndarray
from torch import nn, Tensor
from typing import Tuple, Union
from .block import DqnConv, FcBody, LinearHead, NetworkBlock
from ..prelude import NetFn
from ..utils import Device
from ..utils.misc import iter_prod


class ValuePredictor(ABC):
    @abstractmethod
    def action_values(self, state: ndarray, nostack: bool = False) -> Tensor:
        pass

    @property
    @abstractmethod
    def state_dim(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def action_dim(self) -> int:
        pass


class ValueNet(ValuePredictor, nn.Module):
    """State -> [Value..]
    """
    def __init__(self, body: NetworkBlock, head: NetworkBlock, device: Device = Device()) -> None:
        assert body.output_dim == iter_prod(head.input_dim), \
            'body output and head input must have a same dimention'
        super(ValueNet, self).__init__()
        self.head = head
        if device.is_multi_gpu():
            self.body = device.data_parallel(body)
        else:
            self.body = body
        self.device = device
        self.to(self.device.unwrapped)

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
    def state_dim(self) -> Tuple[int, ...]:
        return self.body.input_dim

    @property
    def action_dim(self) -> int:
        return self.head.output_dim


def dqn_conv(*args, **kwargs) -> NetFn:
    def _net(state_dim: Tuple[int, int, int], action_dim: int, device: Device) -> ValueNet:
        body = DqnConv(state_dim, *args, **kwargs)
        head = LinearHead(body.output_dim, action_dim)
        return ValueNet(body, head, device=device)
    return _net  # type: ignore


def fc(*args, **kwargs) -> NetFn:
    def _net(state_dim: Tuple[int, ...], action_dim: int, device: Device) -> ValueNet:
        body = FcBody(state_dim[0], *args, **kwargs)
        head = LinearHead(body.output_dim, action_dim)
        return ValueNet(body, head, device=device)
    return _net
