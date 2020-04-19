"""Networks for Option-Critic families.
"""
from abc import ABC, abstractmethod
from typing import Sequence, Tuple, Type

from torch import Tensor, nn

from ..prelude import ArrayLike
from ..utils import Device
from .actor_critic import policy_init
from .block import CNNBody, FCBody, LinearHead, NetworkBlock
from .init import Initializer
from .policy import (
    BernoulliDist,
    BernoulliPolicy,
    CategoricalDist,
    CategoricalPolicy,
    Policy,
    PolicyDist,
)
from .prelude import NetFn


class OptionCriticNet(nn.Module, ABC):
    """Network for option critic
    """

    has_mu: bool
    num_options: int
    state_dim: Sequence[int]

    @abstractmethod
    def value(self, states: ArrayLike) -> Tensor:
        pass

    @abstractmethod
    def beta(self, states: ArrayLike) -> BernoulliPolicy:
        pass

    @abstractmethod
    def value_and_beta(self, states: ArrayLike) -> Tuple[Tensor, BernoulliPolicy]:
        pass

    @abstractmethod
    def forward(self, states: ArrayLike) -> Tuple[Policy, Tensor, BernoulliPolicy]:
        pass


class SharedBodyOCNet(OptionCriticNet):
    """An Option Critic Net with shared body and separate π/β/Value heads
    """

    def __init__(
        self,
        body: NetworkBlock,
        actor_head: NetworkBlock,
        value_head: NetworkBlock,
        beta_head: NetworkBlock,
        policy_dist: PolicyDist,
        device: Device = Device(),
    ) -> None:
        super().__init__()
        self.has_mu = False
        self.body = body
        self.actor_head = actor_head
        self.value_head = value_head
        self.beta_head = beta_head
        self.policy_dist = policy_dist
        self.beta_dist = BernoulliDist()
        self.num_options = value_head.output_dim
        self.action_dim = actor_head.output_dim // self.num_options
        self.device = device
        self.state_dim = self.body.input_dim
        self.to(device.unwrapped)

    def value(self, states: ArrayLike) -> Tensor:
        feature = self.body(self.device.tensor(states))
        return self.value_head(feature)

    def beta(self, states: ArrayLike) -> BernoulliPolicy:
        feature = self.body(self.device.tensor(states))
        return self.beta_dist(self.beta_head(feature))

    def value_and_beta(self, states: ArrayLike) -> Tuple[Tensor, BernoulliPolicy]:
        feature = self.body(self.device.tensor(states))
        return self.value_head(feature), self.beta_dist(self.beta_head(feature))

    def forward(self, states: ArrayLike) -> Tuple[Policy, Tensor, BernoulliPolicy]:
        feature = self.body(self.device.tensor(states))
        policy = self.actor_head(feature).view(-1, self.num_options, self.action_dim)
        value = self.value_head(feature)
        beta = self.beta_dist(self.beta_head(feature))
        return self.policy_dist(policy), value, beta


class SharedBodyOCNetWithMu(SharedBodyOCNet):
    """An Option Critic Net with option policy
    """

    def __init__(
        self,
        body: NetworkBlock,
        actor_head: NetworkBlock,
        value_head: NetworkBlock,
        beta_head: NetworkBlock,
        policy_dist: PolicyDist,
        mu_head: NetworkBlock,
        device: Device = Device(),
    ) -> None:
        super().__init__(body, actor_head, value_head, beta_head, policy_dist, device)
        self.has_mu = True
        self.mu_head = mu_head
        self.mu_dist = CategoricalDist(value_head.output_dim)
        self.to(device.unwrapped)

    def forward(
        self, states: ArrayLike
    ) -> Tuple[Policy, Tensor, BernoulliPolicy, CategoricalPolicy]:
        feature = self.body(self.device.tensor(states))
        policy = self.actor_head(feature).view(-1, self.num_options, self.action_dim)
        value = self.value_head(feature)
        beta = self.beta_dist(self.beta_head(feature))
        mu = self.mu_dist(self.mu_head(feature))
        return self.policy_dist(policy), value, beta, mu


def conv_shared(
    num_options: int = 8,
    policy: Type[PolicyDist] = CategoricalDist,
    hidden_channels: Tuple[int, int, int] = (32, 64, 32),
    feature_dim: int = 256,
    has_mu: bool = False,
    beta_init: Initializer = Initializer(),
    **kwargs,
) -> NetFn:
    def _net(
        state_dim: Tuple[int, int, int], action_dim: int, device: Device
    ) -> SharedBodyOCNet:
        body = CNNBody(
            state_dim, hidden_channels=hidden_channels, output_dim=feature_dim, **kwargs
        )
        ac_head = LinearHead(body.output_dim, action_dim * num_options, policy_init())
        value_head = LinearHead(body.output_dim, num_options)
        beta_head = LinearHead(body.output_dim, num_options, init=beta_init)
        dist = policy(action_dim, device)
        if has_mu:
            mu_head = LinearHead(body.output_dim, num_options, policy_init())
            return SharedBodyOCNetWithMu(
                body, ac_head, value_head, beta_head, dist, mu_head, device
            )
        else:
            return SharedBodyOCNet(body, ac_head, value_head, beta_head, dist, device)

    return _net  # type: ignore


def fc_shared(
    num_options: int = 8,
    policy: Type[PolicyDist] = CategoricalDist,
    has_mu: bool = False,
    **kwargs,
) -> NetFn:
    def _net(
        state_dim: Sequence[int], action_dim: int, device: Device
    ) -> SharedBodyOCNet:
        body = FCBody(state_dim[0], **kwargs)
        ac_head = LinearHead(body.output_dim, action_dim * num_options, policy_init())
        value_head = LinearHead(body.output_dim, num_options)
        beta_head = LinearHead(body.output_dim, num_options)
        dist = policy(action_dim, device)
        if has_mu:
            mu_head = LinearHead(body.output_dim, num_options, policy_init())
            return SharedBodyOCNetWithMu(
                body, ac_head, value_head, beta_head, dist, mu_head, device
            )
        else:
            return SharedBodyOCNet(body, ac_head, value_head, beta_head, dist, device)

    return _net  # type: ignore
