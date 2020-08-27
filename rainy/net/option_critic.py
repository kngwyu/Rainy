"""Networks for Option-Critic families.
"""
from abc import ABC, abstractmethod
from typing import Optional, Sequence, Tuple, Type

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
    def qo(self, states: ArrayLike) -> Tensor:
        """ Returns Qo(states, ・), of which the shape is batch_size x n_options.
        """
        pass

    @abstractmethod
    def beta(self, states: ArrayLike) -> BernoulliPolicy:
        """ Returns β(states, ・), of which the shape is batch_size x n_options.
        """
        pass

    @abstractmethod
    def qo_and_beta(self, states: ArrayLike) -> Tuple[Tensor, BernoulliPolicy]:
        """ Returns Qo(states, ・) and β(states, ・).
        """
        pass


class SharedBodyOCNet(OptionCriticNet):
    """ OptionCriticNet with shared body and separate π, Qo and β heads.
    """

    def __init__(
        self,
        body: NetworkBlock,
        action_dim: int,
        num_options: int,
        policy_dist: PolicyDist,
        init: Initializer = Initializer(),
        beta_init: Optional[Initializer] = None,
        policy_init: Initializer = policy_init(),
        device: Device = Device(),
    ) -> None:
        super().__init__()
        self.has_mu = False
        self.body = body
        self.actor_head = LinearHead(
            body.output_dim, num_options * action_dim, init=policy_init
        )
        self.qo_head = LinearHead(body.output_dim, num_options, init=init)
        self.beta_head = LinearHead(
            body.output_dim, num_options, init=beta_init or init
        )
        self.policy_dist = policy_dist
        self.beta_dist = BernoulliDist()
        self.num_options = num_options
        self.action_dim = action_dim
        self.device = device
        self.state_dim = self.body.input_dim
        self.to(device.unwrapped)

    def qo(self, states: ArrayLike) -> Tensor:
        feature = self.body(self.device.tensor(states))
        return self.qo_head(feature)

    def beta(self, states: ArrayLike) -> BernoulliPolicy:
        feature = self.body(self.device.tensor(states))
        return self.beta_dist(self.beta_head(feature))

    def qo_and_beta(self, states: ArrayLike) -> Tuple[Tensor, BernoulliPolicy]:
        feature = self.body(self.device.tensor(states))
        return self.qo_head(feature), self.beta_dist(self.beta_head(feature))

    def forward(self, states: ArrayLike) -> Tuple[Policy, Tensor, BernoulliPolicy]:
        """ Returns π(states), Qo(states, ・) and β(states, ・).
        """
        feature = self.body(self.device.tensor(states))
        policy = self.actor_head(feature).view(-1, self.num_options, self.action_dim)
        qo = self.qo_head(feature)
        beta = self.beta_dist(self.beta_head(feature))
        return self.policy_dist(policy), qo, beta


class SharedBodyOCNetWithMu(SharedBodyOCNet):
    """
    OptionCriticNet with shared body and separate π, Qo, β and μ heads.
    """

    def __init__(
        self,
        body: NetworkBlock,
        action_dim: int,
        num_options: int,
        policy_dist: PolicyDist,
        init: Initializer = Initializer(),
        beta_init: Optional[Initializer] = None,
        policy_init: Initializer = policy_init(),
        device: Device = Device(),
    ) -> None:
        super().__init__(
            body,
            action_dim,
            num_options,
            policy_dist,
            init=init,
            beta_init=beta_init,
            policy_init=policy_init,
            device=device,
        )
        self.mu_head = LinearHead(body.output_dim, num_options, init=policy_init)
        self.mu_dist = CategoricalDist(num_options)
        self.to(device.unwrapped)

    def forward(
        self, states: ArrayLike
    ) -> Tuple[Policy, Tensor, BernoulliPolicy, CategoricalPolicy]:
        """ Returns π(states), Qo(states, ・), β(states, ・) and μ(states).
        """
        feature = self.body(self.device.tensor(states))
        policy = self.actor_head(feature).view(-1, self.num_options, self.action_dim)
        qo = self.qo_head(feature)
        beta = self.beta_dist(self.beta_head(feature))
        mu = self.mu_dist(self.mu_head(feature))
        return self.policy_dist(policy), qo, beta, mu


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
        dist = policy(action_dim, device, noptions=num_options)
        if has_mu:
            cls = SharedBodyOCNetWithMu
        else:
            cls = SharedBodyOCNet
        return cls(
            body, action_dim, num_options, dist, beta_init=beta_init, device=device
        )

    return _net  # type: ignore


def fc_shared(
    num_options: int = 8,
    policy: Type[PolicyDist] = CategoricalDist,
    has_mu: bool = False,
    beta_init: Initializer = Initializer(),
    **kwargs,
) -> NetFn:
    def _net(
        state_dim: Sequence[int], action_dim: int, device: Device
    ) -> SharedBodyOCNet:
        body = FCBody(state_dim[0], **kwargs)
        dist = policy(action_dim, device, noptions=num_options)
        if has_mu:
            cls = SharedBodyOCNetWithMu
        else:
            cls = SharedBodyOCNet
        return cls(
            body, action_dim, num_options, dist, beta_init=beta_init, device=device
        )

    return _net  # type: ignore
