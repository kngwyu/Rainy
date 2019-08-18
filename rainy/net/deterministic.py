from abc import ABC, abstractmethod
from rainy.utils import Device
import torch
from torch import nn, Tensor
from typing import Sequence, Tuple, Union
from .block import FcBody, LinearHead, NetworkBlock
from .init import Initializer
from .prelude import NetFn
from .value import ContinuousQFunction
from ..prelude import Array, Self


class DeterministicPolicyNet(ABC):
    @abstractmethod
    def action(self, state: Union[Array, Tensor]) -> Tensor:
        pass


class SoftUpdate(nn.Module):
    @torch.no_grad()
    def soft_update(self, other: Self, coef: float) -> None:
        for s_param, o_param in zip(self.parameters(), other.parameters()):
            s_param.copy_(s_param * (1.0 - coef) + o_param * coef)


class DdpgNet(SoftUpdate, ContinuousQFunction, DeterministicPolicyNet):
    pass


class SeparatedDdpgNet(DdpgNet):
    def __init__(
            self,
            actor_body: NetworkBlock,
            critic_body: NetworkBlock,
            action_dim: int,
            init: Initializer = Initializer(scale=1.0e-3),
            device: Device = Device(),
    ) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            actor_body,
            LinearHead(actor_body.output_dim, action_dim, init=init),
            nn.Tanh(),
        )
        self.critic = nn.Sequential(
            critic_body,
            LinearHead(critic_body.output_dim, 1, init=init)
        )
        self.to(device.unwrapped)
        self.device = device

    def action(self, states: Union[Array, Tensor]) -> Tensor:
        s = self.device.tensor(states)
        return self.actor(s)

    def q_value(self, states: Union[Array, Tensor], action: Union[Array, Tensor]) -> Tensor:
        s = self.device.tensor(states)
        a = self.device.tensor(action)
        sa = torch.cat((s, a), dim=1)
        return self.critic(sa)

    def forward(
            self,
            states: Union[Array, Tensor],
            action: Union[Array, Tensor]
    ) -> Tuple[Tensor, Tensor]:
        s = self.device.tensor(states)
        a = self.device.tensor(action)
        sa = torch.cat((s, a), dim=1)
        return self.actor(s), self.critic(sa)


def fc_seprated(
        actor_units: Sequence[int] = [400, 300],
        critic_units: Sequence[int] = [400, 300],
) -> NetFn:
    """FC body head ActorCritic network
    """
    def _net(state_dim: Tuple[int, ...], action_dim: int, device: Device) -> SeparatedDdpgNet:
        actor_body = FcBody(state_dim[0], units=actor_units)
        critic_body = FcBody(state_dim[0] + action_dim, units=critic_units)
        return SeparatedDdpgNet(actor_body, critic_body, action_dim, device=device)
    return _net
