from numpy import ndarray
from torch import nn, Tensor
from typing import Callable
from .body import NetworkBody
from .head import NetworkHead
from ..lib import Device

# TODO: is it enough robust?
class ValueNet(nn.Module):
    """State -> [Value..]
    """

    def __init__(
            self,
            body: NetworkBody,
            head: NetworkHead,
            device: Device = Device(),
    ) -> None:
        assert body.output_dim == head.input__dim, \
            'body output and head input must have a same dimention'
        self.head = head
        self.body = body
        self.device = device

    @property
    def state_dim(self) -> int:
        self.body.input_dim

    @property
    def action_dim(self) -> int:
        self.head.output_dim

    def action_values(self, state: ndarray) -> Tensor:
        x = self.device.tensor(state)
        x = self.body(x)
        x = self.head(x)
        return x


class ValueNetGen:
    """Value Net builder
    """
    def __init__(
            self,
            body_gen: Callable[[int], NetworkBody],
            head_gen: Callable[[int, int], NetworkHead],
    ) -> None:
        self.body_gen = body_gen
        self.head_gen = head_gen

    def __call__(
            self,
            state_dim: int,
            action_dim: int,
            device: Device
    ) -> ValueNet:
        body = self.body_gen(state_dim)
        head_input_dim = body.output_dim
        head = self.head_gen(head_input_dim, action_dim)
        return ValueNet(body, head, device=device)

