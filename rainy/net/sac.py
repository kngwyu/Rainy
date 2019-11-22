import copy
import itertools
import torch
from torch import nn, Tensor
from typing import Iterable, List, Sequence, Tuple
from .block import FcBody, LinearHead, NetworkBlock
from .init import Initializer, fanin_uniform, constant
from .misc import SoftUpdate
from .policy import Policy, PolicyDist, TanhGaussianDist
from .prelude import NetFn
from .value import ContinuousQFunction
from ..prelude import ArrayLike, Self
from ..utils import Device


class SACTarget(SoftUpdate):
    def __init__(self, critic1: nn.Module, critic2: nn.Module, device: Device) -> None:
        super().__init__()
        self.critic1 = critic1
        self.critic2 = critic2
        self.device = device

    def soft_update(self, other: Self, coef: float) -> None:
        SoftUpdate.soft_update(self.critic1, other.critic1, coef)  # type: ignore
        SoftUpdate.soft_update(self.critic2, other.critic2, coef)  # type: ignore

    def q_values(self, states: ArrayLike, action: ArrayLike) -> Tuple[Tensor, Tensor]:
        sa = torch.cat((self.device.tensor(states), self.device.tensor(action)), dim=1)
        return self.critic1(sa), self.critic2(sa)


class SeparatedSACNet(nn.Module, ContinuousQFunction):
    def __init__(
        self,
        actor_body: NetworkBlock,
        critic_body1: NetworkBlock,
        critic_body2: NetworkBlock,
        policy_dist: PolicyDist,
        device: Device = Device(),
        init: Initializer = Initializer(
            weight_init=fanin_uniform(), bias_init=constant(0.1)
        ),
    ) -> None:
        super().__init__()
        self.actor = nn.Sequential(
            actor_body,
            LinearHead(actor_body.output_dim, policy_dist.input_dim, init=init),
        )
        self.critic1 = nn.Sequential(
            critic_body1, LinearHead(critic_body1.output_dim, 1, init=init)
        )
        self.critic2 = nn.Sequential(
            critic_body2, LinearHead(critic_body2.output_dim, 1, init=init)
        )
        self.policy_dist = policy_dist
        self.device = device
        self.to(device.unwrapped)

    def q_value(self, states: ArrayLike, action: ArrayLike) -> Tensor:
        sa = torch.cat((self.device.tensor(states), self.device.tensor(action)), dim=1)
        return self.critic1(sa)

    def q_values(self, states: ArrayLike, action: ArrayLike) -> Tuple[Tensor, Tensor]:
        sa = torch.cat((self.device.tensor(states), self.device.tensor(action)), dim=1)
        return self.critic1(sa), self.critic2(sa)

    def get_target(self) -> SACTarget:
        return SACTarget(
            copy.deepcopy(self.critic1), copy.deepcopy(self.critic2), self.device
        )

    def policy(self, states: ArrayLike) -> Policy:
        st = self.device.tensor(states)
        if st.dim() == 1:
            st = st.view(1, -1)
        policy_params = self.actor(st)
        return self.policy_dist(policy_params)

    def actor_params(self) -> Iterable[Tensor]:
        return self.actor.parameters()

    def critic_params(self) -> Iterable[Tensor]:
        return itertools.chain(self.critic1.parameters(), self.critic2.parameters())

    def forward(
        self, states: ArrayLike, action: ArrayLike
    ) -> Tuple[Tensor, Tensor, Policy]:
        s, a = self.device.tensor(states), self.device.tensor(action)
        sa = torch.cat((s, a), dim=1)
        q1, q2 = self.critic1(sa), self.critic2(sa)
        policy = self.policy_dist(self.actor(s))
        return q1, q2, policy


def fc_separated(
    actor_units: List[int] = [256, 256],
    critic_units: List[int] = [256, 256],
    policy_type: type = TanhGaussianDist,
    init: Initializer = Initializer(
        weight_init=fanin_uniform(), bias_init=constant(0.1)
    ),
) -> NetFn:
    """SAC network with separated bodys
    """

    def _net(
        state_dim: Sequence[int], action_dim: int, device: Device
    ) -> SeparatedSACNet:
        actor_body = FcBody(state_dim[0], units=actor_units, init=init)
        critic1 = FcBody(state_dim[0] + action_dim, units=critic_units, init=init)
        critic2 = FcBody(state_dim[0] + action_dim, units=critic_units, init=init)
        policy = policy_type(action_dim)
        return SeparatedSACNet(
            actor_body, critic1, critic2, policy, device=device, init=init,
        )

    return _net
