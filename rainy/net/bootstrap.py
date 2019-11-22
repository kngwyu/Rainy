from abc import abstractmethod
import numpy as np
from torch import nn, Tensor
from typing import List, Sequence
from .block import FcBody, LinearHead, NetworkBlock
from .prelude import NetFn
from .value import DiscreteQFunction, DiscreteQValueNet
from ..utils import Device
from ..prelude import Array, ArrayLike


class BootstrappedQFunction(DiscreteQFunction):
    active_head: int

    @abstractmethod
    def forward(self, index: int, x: ArrayLike) -> Tensor:
        pass


class SeparatedBootQValueNet(BootstrappedQFunction, nn.Module):
    def __init__(self, q_nets: List[DiscreteQFunction]):
        super().__init__()
        self.q_nets = nn.ModuleList(q_nets)
        self.active_head = 0

    def q_value(self, state: Array) -> Tensor:
        return self.q_nets[self.active_head].q_value(state)

    def forward(self, index: int, x: ArrayLike) -> Tensor:
        return self.q_nets[index](x)

    @property
    def state_dim(self) -> Sequence[int]:
        return self.q_nets[0].state_dim

    @property
    def action_dim(self) -> int:
        return self.q_nets[0].action_dim


class SharedBootQValueNet(DiscreteQFunction, nn.Module):
    """State -> [Value..]
    """

    def __init__(
        self, body: NetworkBlock, heads: List[NetworkBlock], device: Device = Device()
    ) -> None:
        if body.output_dim != np.prod(head[0].input_dim):
            raise ValueError("body output and head input must have a same dimention")
        super().__init__()
        self.body = body
        self.head = nn.ModuleList(head)
        self.device = device
        self.to(self.device.unwrapped)

    def forward(self, index: int, x: ArrayLike) -> Tensor:
        raise NotImplementedError()

    @property
    def state_dim(self) -> Sequence[int]:
        return self.body.input_dim

    @property
    def action_dim(self) -> int:
        return self.head[0].output_dim


def fc_separated(n_ensembles: int, *args, **kwargs) -> NetFn:
    def _net(
        state_dim: Sequence[int], action_dim: int, device: Device
    ) -> SeparatedBootQValueNet:
        q_nets = []
        for _ in range(n_ensembles):
            body = FcBody(state_dim[0], *args, **kwargs)
            head = LinearHead(body.output_dim, action_dim)
            q_nets.append(DiscreteQValueNet(body, head, device=device))
        return SeparatedBootQValueNet(q_nets)

    return _net
