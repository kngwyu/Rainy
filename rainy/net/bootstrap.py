from abc import abstractmethod
from copy import deepcopy
from typing import List, Sequence

import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..prelude import Array, ArrayLike
from ..utils import Device
from .block import FCBody, LinearHead, NetworkBlock
from .init import Initializer, xavier_uniform
from .prelude import NetFn
from .value import DiscreteQFunction, DiscreteQValueNet


class BootstrappedQFunction(DiscreteQFunction):
    device: Device

    @abstractmethod
    def forward(self, states: ArrayLike) -> Tensor:
        pass

    @abstractmethod
    def q_i_s(self, index: int, states: ArrayLike) -> Tensor:
        pass

    def q_value(self, state: Array) -> Tensor:
        values = self(state)
        return values.mean(dim=1)

    def q_s_a(self, states: ArrayLike, actions: ArrayLike) -> Tensor:
        qs = self(self.device.tensor(states))
        act = self.device.tensor(actions, dtype=torch.long)
        action_mask = F.one_hot(act, num_classes=qs.size(-1)).float()
        return torch.einsum("bka,ba->bk", qs, action_mask)


class SeparatedBootQValueNet(BootstrappedQFunction, nn.Module):
    def __init__(self, q_nets: List[DiscreteQFunction]):
        super().__init__()
        self.q_nets = nn.ModuleList(q_nets)
        self.device = q_nets[0].device

    def forward(self, x: ArrayLike) -> Tensor:
        return torch.stack([q(x) for q in self.q_nets], dim=1)

    def q_i_s(self, index: int, states: ArrayLike) -> Tensor:
        return self.q_nets[index](states)

    @property
    def state_dim(self) -> Sequence[int]:
        return self.q_nets[0].state_dim

    @property
    def action_dim(self) -> int:
        return self.q_nets[0].action_dim


class SharedBootQValueNet(DiscreteQFunction, nn.Module):
    """INCOMPLETE
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


class RPFQValueNet(DiscreteQFunction, nn.Module):
    """QValueNet with Randomized Prior function(https://arxiv.org/abs/1806.03335)
    """

    def __init__(
        self,
        body: NetworkBlock,
        head: NetworkBlock,
        prior_scale: float = 1.0,
        device: Device = Device(),
        init: Initializer = Initializer(),
    ) -> None:
        if body.output_dim != np.prod(head.input_dim):
            raise ValueError("body output and head input must have a same dimention")
        super().__init__()
        self.model = nn.Sequential(body, head)
        self.prior = init(deepcopy(self.model))
        self.device = device
        self.prior_scale = prior_scale
        self.to(self.device.unwrapped)

    def q_value(self, state: Array, nostack: bool = False) -> Tensor:
        if nostack:
            return self.forward(state)
        else:
            return self.forward(np.stack([state]))

    def forward(self, x: ArrayLike) -> Tensor:
        x = self.device.tensor(x)
        raw = self.model(x)
        with torch.no_grad():
            prior = self.prior(x)
        return raw.add_(prior.mul_(self.prior_scale))

    @property
    def state_dim(self) -> Sequence[int]:
        return self.model[0].input_dim

    @property
    def action_dim(self) -> int:
        return self.model[1].output_dim


def fc_separated(n_ensembles: int, *args, **kwargs) -> NetFn:
    def _net(
        state_dim: Sequence[int], action_dim: int, device: Device
    ) -> SeparatedBootQValueNet:
        q_nets = []
        for _ in range(n_ensembles):
            body = FCBody(state_dim[0], *args, **kwargs)
            head = LinearHead(body.output_dim, action_dim)
            q_nets.append(DiscreteQValueNet(body, head, device=device))
        return SeparatedBootQValueNet(q_nets)

    return _net


def rpf_fc_separated(
    n_ensembles: int,
    prior_scale: float = 1.0,
    init: Initializer = Initializer(weight_init=xavier_uniform()),
    **kwargs,
) -> NetFn:
    def _net(
        state_dim: Sequence[int], action_dim: int, device: Device
    ) -> SeparatedBootQValueNet:
        q_nets = []
        for _ in range(n_ensembles):
            body = FCBody(state_dim[0], init=init, **kwargs)
            head = LinearHead(body.output_dim, action_dim, init=init)
            prior_model = RPFQValueNet(body, head, prior_scale, init=init)
            q_nets.append(prior_model)
        return SeparatedBootQValueNet(q_nets)

    return _net
