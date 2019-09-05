from abc import ABC, abstractmethod
from rainy.utils import Device
import torch
from torch import nn, Tensor
from typing import Sequence, Tuple, Union
from .block import FcBody, LinearHead, NetworkBlock
from .init import Initializer, kaiming_uniform
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
            action_coef: float = 1.0,
            device: Device = Device(),
            init: Initializer = Initializer(weight_init=kaiming_uniform(a=3 ** 0.5)),
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
        self.action_coef = action_coef
        self.device = device

    def action(self, states: Union[Array, Tensor]) -> Tensor:
        s = self.device.tensor(states)
        return self.actor(s).mul(self.action_coef)

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
        return self.actor(s).mul(self.action_coef), self.critic(sa)


class SeparatedTd3Net(SeparatedDdpgNet):
    def __init__(
            self,
            actor_body: NetworkBlock,
            critic_body: NetworkBlock,
            critic_body2: NetworkBlock,
            action_dim: int,
            action_coef: float = 1.0,
            device: Device = Device(),
            init: Initializer = Initializer(weight_init=kaiming_uniform(a=3 ** 0.5)),
    ) -> None:
        super().__init__(
            actor_body,
            critic_body,
            action_dim,
            action_coef=action_coef,
            device=device,
            init=init,
        )
        self.critic2 = nn.Sequential(
            critic_body2,
            LinearHead(critic_body2.output_dim, 1, init=init)
        )
        self.to(device.unwrapped)

    def q_values(self, states: Union[Array, Tensor], action: Union[Array, Tensor]) -> Tensor:
        s = self.device.tensor(states)
        a = self.device.tensor(action)
        sa = torch.cat((s, a), dim=1)
        return self.critic(sa), self.critic2(sa)


def fc_seprated(
        action_coef: float = 1.0,
        actor_units: Sequence[int] = [400, 300],
        critic_units: Sequence[int] = [400, 300],
        init: Initializer = Initializer(weight_init=kaiming_uniform(a=3 ** 0.5)),
) -> NetFn:
    """FC body head ActorCritic network
    """
    def _net(state_dim: Tuple[int, ...], action_dim: int, device: Device) -> SeparatedDdpgNet:
        actor_body = FcBody(state_dim[0], units=actor_units, init=init)
        critic_body = FcBody(state_dim[0] + action_dim, units=critic_units, init=init)
        return SeparatedDdpgNet(
            actor_body,
            critic_body,
            action_dim,
            action_coef=action_coef,
            device=device,
            init=init,
        )
    return _net


def td3_fc_seprated(
        action_coef: float = 1.0,
        actor_units: Sequence[int] = [400, 300],
        critic_units: Sequence[int] = [400, 300],
        init: Initializer = Initializer(weight_init=kaiming_uniform(a=3 ** 0.5)),
) -> NetFn:
    """FC body head ActorCritic network
    """
    def _net(state_dim: Tuple[int, ...], action_dim: int, device: Device) -> SeparatedDdpgNet:
        actor_body = FcBody(state_dim[0], units=actor_units, init=init)
        critic1 = FcBody(state_dim[0] + action_dim, units=critic_units, init=init)
        critic2 = FcBody(state_dim[0] + action_dim, units=critic_units, init=init)
        return SeparatedTd3Net(
            actor_body,
            critic1,
            critic2,
            action_dim,
            action_coef=action_coef,
            device=device,
            init=init,
        )
    return _net
