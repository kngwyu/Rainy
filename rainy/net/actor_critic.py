from abc import ABC, abstractmethod
import numpy as np
from torch import nn, Tensor
from typing import Callable, List, Optional, Sequence, Tuple
from .block import DQNConv, FcBody, ResNetBody, LinearHead, NetworkBlock
from .init import Initializer, orthogonal
from .policy import CategoricalDist, Policy, PolicyDist
from .prelude import NetFn
from .recurrent import DummyRnn, RnnBlock, RnnState
from ..prelude import ArrayLike
from ..utils import Device


class ActorCriticNet(nn.Module, ABC):
    """Network with Policy + Value Head
    """

    @property
    @abstractmethod
    def recurrent_body(self) -> RnnBlock:
        pass

    @property
    def is_recurrent(self) -> bool:
        return not isinstance(self.recurrent_body, DummyRnn)

    @abstractmethod
    def policy(
        self,
        states: ArrayLike,
        rnns: Optional[RnnState] = None,
        masks: Optional[Tensor] = None,
    ) -> Tuple[Policy, RnnState]:
        pass

    @abstractmethod
    def value(
        self,
        states: ArrayLike,
        rnns: Optional[RnnState] = None,
        masks: Optional[Tensor] = None,
    ) -> Tensor:
        pass

    @abstractmethod
    def forward(
        self,
        states: ArrayLike,
        rnns: Optional[RnnState] = None,
        masks: Optional[Tensor] = None,
    ) -> Tuple[Policy, Tensor, RnnState]:
        pass


class SharedBodyACNet(ActorCriticNet):
    """An Actor Critic network with common body + separate value/policy heads.
    Basically it's same as the one used in the Atari experimtent in the A3C paper.
    """

    def __init__(
        self,
        body: NetworkBlock,
        actor_head: NetworkBlock,  # policy
        critic_head: NetworkBlock,  # value
        policy_dist: PolicyDist,
        recurrent_body: RnnBlock = DummyRnn(),
        device: Device = Device(),
    ) -> None:
        assert body.output_dim == np.prod(
            actor_head.input_dim
        ), "body output and action_head input must have a same dimention"
        assert body.output_dim == np.prod(
            critic_head.input_dim
        ), "body output and action_head input must have a same dimention"
        super().__init__()
        self.body = body
        self.device = device
        self.body = body
        self.actor_head = actor_head
        self.critic_head = critic_head
        self.policy_dist = policy_dist
        self._rnn_body = recurrent_body
        self.to(device.unwrapped)

    @property
    def state_dim(self) -> Sequence[int]:
        return self.body.input_dim

    @property
    def action_dim(self) -> int:
        return self.actor_head.output_dim

    @property
    def recurrent_body(self) -> RnnBlock:
        return self._rnn_body

    def _features(
        self,
        states: ArrayLike,
        rnns: Optional[RnnState] = None,
        masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, RnnState]:
        res = self.body(self.device.tensor(states))
        if rnns is None:
            rnns = self._rnn_body.initial_state(res.size(0), self.device)
        res = self._rnn_body(res, rnns, masks)
        return res

    def policy(
        self,
        states: ArrayLike,
        rnns: Optional[RnnState] = None,
        masks: Optional[Tensor] = None,
    ) -> Tuple[Policy, RnnState]:
        features, rnn_next = self._features(states, rnns, masks)
        return self.policy_dist(self.actor_head(features)), rnn_next

    def value(
        self,
        states: ArrayLike,
        rnns: Optional[RnnState] = None,
        masks: Optional[Tensor] = None,
    ) -> Tensor:
        features = self._features(states, rnns, masks)[0]
        return self.critic_head(features).squeeze()

    def forward(
        self,
        states: ArrayLike,
        rnns: Optional[RnnState] = None,
        masks: Optional[Tensor] = None,
    ) -> Tuple[Policy, Tensor, RnnState]:
        features, rnn_next = self._features(states, rnns, masks)
        policy, value = self.actor_head(features), self.critic_head(features)
        return self.policy_dist(policy), value.squeeze(), rnn_next


def policy_init() -> Initializer:
    """Use small value for policy layer to make policy entroy larger
    """
    return Initializer(weight_init=orthogonal(0.01))


def _make_ac_shared(
    body: NetworkBlock,
    policy_dist: PolicyDist,
    device: Device,
    rnn: Callable[[int, int], RnnBlock],
) -> SharedBodyACNet:
    rnn_ = rnn(body.output_dim, body.output_dim)
    ac_head = LinearHead(body.output_dim, policy_dist.input_dim, policy_init())
    cr_head = LinearHead(body.output_dim, 1)
    return SharedBodyACNet(
        body, ac_head, cr_head, policy_dist, recurrent_body=rnn_, device=device
    )


def ac_conv(
    policy: Callable[[int, Device], PolicyDist] = CategoricalDist,
    hidden_channels: Tuple[int, int, int] = (32, 64, 32),
    output_dim: int = 256,
    rnn: Callable[[int, int], RnnBlock] = DummyRnn,
    **kwargs
) -> NetFn:
    """Convolutuion network used for atari experiments
       in A3C paper(https://arxiv.org/abs/1602.01783)
    """

    def _net(
        state_dim: Tuple[int, int, int], action_dim: int, device: Device
    ) -> SharedBodyACNet:
        body = DQNConv(
            state_dim, hidden_channels=hidden_channels, output_dim=output_dim, **kwargs
        )
        policy_dist = policy(action_dim, device)
        return _make_ac_shared(body, policy_dist, device, rnn)

    return _net  # type: ignore


def fc_shared(
    policy: Callable[[int, Device], PolicyDist] = CategoricalDist,
    rnn: Callable[[int, int], RnnBlock] = DummyRnn,
    **kwargs
) -> NetFn:
    """FC body head ActorCritic network
    """

    def _net(
        state_dim: Sequence[int], action_dim: int, device: Device
    ) -> SharedBodyACNet:
        body = FcBody(state_dim[0], **kwargs)
        policy_dist = policy(action_dim, device)
        return _make_ac_shared(body, policy_dist, device, rnn)

    return _net


def impala_conv(
    policy: Callable[[int, Device], PolicyDist] = CategoricalDist,
    channels: List[int] = [16, 32, 32],
    rnn: Callable[[int, int], RnnBlock] = DummyRnn,
    **kwargs
) -> NetFn:
    """Convolutuion network used in IMPALA
    """

    def _net(
        state_dim: Tuple[int, int, int], action_dim: int, device: Device
    ) -> SharedBodyACNet:
        body = ResNetBody(state_dim, channels, **kwargs)
        policy_dist = policy(action_dim, device)
        return _make_ac_shared(body, policy_dist, device, rnn)

    return _net  # type: ignore
