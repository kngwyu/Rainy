"""Networks for termination critics.
"""

from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from typing import NamedTuple, Sequence, Tuple, Type
from .actor_critic import policy_init
from .block import CNNBody, CNNBodyWithoutFc, FcBody, LinearHead, NetworkBlock
from .init import Initializer
from .policy import (
    BernoulliDist,
    BernoulliPolicy,
    CategoricalDist,
    Policy,
    PolicyDist,
)
from .prelude import NetFn
from ..prelude import ArrayLike
from ..utils import Device


class OptionActorCriticNet(nn.Module, ABC):
    """Actor Critic Networks with options
    """

    num_options: int
    state_dim: Sequence[int]

    @abstractmethod
    def opt_q(self, states: ArrayLike) -> Tensor:
        pass

    @abstractmethod
    def opt_pi(self, states: ArrayLike) -> Policy:
        pass

    @abstractmethod
    def forward(self, states: ArrayLike) -> Tuple[Policy, Tensor]:
        pass


class SharedOACNet(OptionActorCriticNet):
    """An Option Critic Net with shared body and separate π/β/Value heads
    """

    def __init__(
        self,
        body: NetworkBlock,
        actor_head: NetworkBlock,
        optq_head: NetworkBlock,
        policy_dist: PolicyDist,
        device: Device = Device(),
    ) -> None:
        super().__init__()
        self.has_mu = False
        self.body = body
        self.actor_head = actor_head
        self.optq_head = optq_head
        self.policy_dist = policy_dist
        self.num_options = optq_head.output_dim
        self.action_dim = actor_head.output_dim // self.num_options
        self.device = device
        self.state_dim = self.body.input_dim
        self.to(device.unwrapped)

    def opt_q(self, states: ArrayLike) -> Tensor:
        feature = self.body(self.device.tensor(states))
        return self.optq_head(feature)

    def opt_pi(self, states: ArrayLike) -> Policy:
        feature = self.body(self.device.tensor(states))
        logits = self.actor_head(feature).view(-1, self.num_options, self.action_dim)
        return self.policy_dist(logits)

    def forward(self, states: ArrayLike) -> Tuple[Policy, Tensor]:
        feature = self.body(self.device.tensor(states))
        logits = self.actor_head(feature).view(-1, self.num_options, self.action_dim)
        opt_q = self.optq_head(feature)
        return self.policy_dist(logits), opt_q


def oac_conv_shared(
    num_options: int = 4,
    policy: Type[PolicyDist] = CategoricalDist,
    hidden_channels: Tuple[int, int, int] = (32, 64, 32),
    feature_dim: int = 256,
    **cnn_args,
) -> NetFn:
    def _net(
        state_dim: Tuple[int, int, int], action_dim: int, device: Device
    ) -> SharedOACNet:
        body = CNNBody(
            state_dim,
            hidden_channels=hidden_channels,
            output_dim=feature_dim,
            **cnn_args,
        )
        ac_head = LinearHead(body.output_dim, action_dim * num_options, policy_init())
        optq_head = LinearHead(body.output_dim, num_options)
        dist = policy(action_dim, device)
        return SharedOACNet(body, ac_head, optq_head, dist, device)

    return _net  # type: ignore


def oac_fc_shared(
    num_options: int = 4, policy: Type[PolicyDist] = CategoricalDist, **fc_args,
) -> NetFn:
    def _net(state_dim: Sequence[int], action_dim: int, device: Device) -> SharedOACNet:
        body = FcBody(state_dim[0], **fc_args)
        ac_head = LinearHead(body.output_dim, action_dim * num_options, policy_init())
        optq_head = LinearHead(body.output_dim, num_options)
        dist = policy(action_dim, device)
        return SharedOACNet(body, ac_head, optq_head, dist, device)

    return _net  # type: ignore


class TCOutput(NamedTuple):
    beta: BernoulliPolicy
    p: Tensor
    p_mu: Tensor
    baseline: Tensor


class TerminationCriticNet(nn.Module, ABC):
    """Termination β and with transition models
    """

    state_dim: Sequence[int]

    @abstractmethod
    def beta(self, xs: ArrayLike, xf: ArrayLike) -> BernoulliPolicy:
        pass

    @abstractmethod
    def forward(self, xs: ArrayLike, xf: ArrayLike) -> TCOutput:
        pass

    @abstractmethod
    def p(self, xs: ArrayLike, xf: ArrayLike) -> Tensor:
        pass


class SharedTCNet(TerminationCriticNet):
    def __init__(
        self,
        body1: NetworkBlock,
        body2: NetworkBlock,
        feature_dim: int,
        num_options: int,
        device: Device = Device(),
        init: Initializer = Initializer(),
    ) -> None:
        super().__init__()
        self.body_xs = body1
        self.body_xf = body2
        feature_in = body1.output_dim + body2.output_dim
        self.feature = LinearHead(feature_in, feature_dim, init=init)
        self.beta_head = nn.Sequential(
            LinearHead(feature_dim, num_options, init=init), BernoulliDist(),
        )
        self.p_head = nn.Sequential(
            LinearHead(feature_dim, num_options, init=init), nn.Sigmoid(),
        )
        self.p_mu_head = nn.Sequential(
            LinearHead(feature_dim, num_options, init=init), nn.Sigmoid(),
        )
        self.baseline_head = LinearHead(feature_dim, num_options, init=init)
        self.device = device
        self.state_dim = body1.input_dim
        self.to(device.unwrapped)

    def beta(self, xs: ArrayLike, xf: ArrayLike) -> BernoulliPolicy:
        return self.beta_head(self._feature(xs, xf))

    def forward(self, xs: ArrayLike, xf: ArrayLike) -> TCOutput:
        feature = self._feature(xs, xf)
        beta = self.beta_head(feature)
        p = self.p_head(feature)
        p_mu_head = self.p_mu_head(feature)
        baseline = self.baseline_head(feature)
        return TCOutput(beta, p, p_mu_head, baseline)

    def _feature(self, xs: ArrayLike, xf: ArrayLike) -> TCOutput:
        xst, xft = self.device.tensor(xs), self.device.tensor(xf)
        batch_size = xst.size(0)
        xs_feature = self.body_xs(xst).view(batch_size, -1)
        xf_feature = self.body_xf(xft).view(batch_size, -1)
        xs_xf = torch.cat((xs_feature, xf_feature), dim=1)
        return self.feature(xs_xf)

    def p(self, xs: ArrayLike, xf: ArrayLike) -> BernoulliPolicy:
        return self.p_head(self._feature(xs, xf))


def tc_conv_shared(
    num_options: int = 4,
    hidden_channels: Tuple[int, int, int] = (32, 64, 32),
    feature_dim: int = 256,
    head_init: Initializer = Initializer(),
    **cnn_args,
) -> NetFn:
    def _net(
        state_dim: Tuple[int, int, int], _action_dim: int, device: Device
    ) -> SharedTCNet:
        body1 = CNNBodyWithoutFc(state_dim, hidden_channels=hidden_channels, **cnn_args)
        body2 = CNNBodyWithoutFc(state_dim, hidden_channels=hidden_channels, **cnn_args)
        return SharedTCNet(
            body1,
            body2,
            num_options=num_options,
            feature_dim=feature_dim,
            device=device,
            init=head_init,
        )

    return _net  # type: ignore


def tc_fc_shared(
    num_options: int = 4,
    feature_dim: int = 256,
    head_init: Initializer = Initializer(),
    **fc_args,
) -> NetFn:
    def _net(state_dim: Sequence[int], _action_dim: int, device: Device) -> SharedTCNet:
        body1 = FcBody(state_dim[0], **fc_args)
        body2 = FcBody(state_dim[0], **fc_args)
        return SharedTCNet(
            body1,
            body2,
            num_options=num_options,
            feature_dim=feature_dim,
            device=device,
            init=head_init,
        )

    return _net  # type: ignore
