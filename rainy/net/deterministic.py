from abc import ABC, abstractmethod
from rainy.utils import Device
from torch import nn
from typing import Union
from .block import FcBody, LinearHead, NetworkBlock
from .init import Initializer
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
            s_param.copy_(s_param * (1.0 - coef) + t_param * coef)


class DdpgACNet(SoftUpdate, ContinuousQFunction, DeterministicPolicyNet):
    def __init__(
            self,
            actor_body: NetworkBlock,
            critic_body: NetworkBlock,
            action_dim: int,
            max_action: float,
            init: Initializer = Initializer(),
            device: Device = Device(),
    ) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            actor_body,
            LinearHead(actor_body.output_dim, action_dim, init=init),
            nn.Tanh(inplace=True)
        )
        self.critic = nn.Sequential(
            critic_body,
            LienarHead(critic_body.output_dim, action_dim, init=init)
        )
        self.max_action = max_action
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
