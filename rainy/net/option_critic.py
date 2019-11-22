from abc import ABC, abstractmethod
from torch import nn, Tensor
from typing import Callable, Sequence, Tuple
from .actor_critic import policy_init
from .block import DQNConv, FcBody, LinearHead, NetworkBlock
from .policy import BernoulliDist, BernoulliPolicy, CategoricalDist, Policy, PolicyDist
from .prelude import NetFn
from ..prelude import ArrayLike
from ..utils import Device


class OptionCriticNet(nn.Module, ABC):
    """Network for option critic
    """

    num_options: int
    state_dim: Sequence[int]

    @abstractmethod
    def opt_q(self, states: ArrayLike) -> Tensor:
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
        optq_head: NetworkBlock,
        beta_head: NetworkBlock,
        policy_dist: PolicyDist,
        device: Device = Device(),
    ) -> None:
        super().__init__()
        self.body = body
        self.actor_head = actor_head
        self.optq_head = optq_head
        self.beta_head = beta_head
        self.policy_dist = policy_dist
        self.beta_dist = BernoulliDist(1)
        self.num_options = optq_head.output_dim
        self.action_dim = actor_head.output_dim // self.num_options
        self.device = device
        self.state_dim = self.body.input_dim
        self.to(device.unwrapped)

    def opt_q(self, states: ArrayLike) -> Tensor:
        feature = self.body(self.device.tensor(states))
        return self.optq_head(feature)

    def forward(self, states: ArrayLike) -> Tuple[Policy, Tensor, BernoulliPolicy]:
        feature = self.body(self.device.tensor(states))
        policy = self.actor_head(feature).view(-1, self.num_options, self.action_dim)
        opt_q = self.optq_head(feature)
        beta = self.beta_dist(self.beta_head(feature))
        return self.policy_dist(policy), opt_q, beta


def conv_shared(
    num_options: int = 8,
    policy: Callable[[int, Device], PolicyDist] = CategoricalDist,
    hidden_channels: Tuple[int, int, int] = (32, 64, 32),
    output_dim: int = 256,
    **kwargs
) -> NetFn:
    def _net(
        state_dim: Tuple[int, int, int], action_dim: int, device: Device
    ) -> SharedBodyOCNet:
        body = DQNConv(
            state_dim, hidden_channels=hidden_channels, output_dim=output_dim, **kwargs
        )
        ac_head = LinearHead(body.output_dim, action_dim * num_options, policy_init())
        optq_head = LinearHead(body.output_dim, num_options)
        beta_head = LinearHead(body.output_dim, num_options)
        dist = policy(action_dim, device)
        return SharedBodyOCNet(body, ac_head, optq_head, beta_head, dist, device)

    return _net  # type: ignore


def fc_shared(
    num_options: int = 8,
    policy: Callable[[int, Device], PolicyDist] = CategoricalDist,
    **kwargs
) -> NetFn:
    def _net(
        state_dim: Sequence[int], action_dim: int, device: Device
    ) -> SharedBodyOCNet:
        body = FcBody(state_dim[0], **kwargs)
        ac_head = LinearHead(body.output_dim, action_dim * num_options, policy_init())
        optq_head = LinearHead(body.output_dim, num_options)
        beta_head = LinearHead(body.output_dim, num_options)
        dist = policy(action_dim, device)
        return SharedBodyOCNet(body, ac_head, optq_head, beta_head, dist, device)

    return _net  # type: ignore
