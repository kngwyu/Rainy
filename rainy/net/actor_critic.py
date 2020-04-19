from abc import ABC, abstractmethod
from typing import List, Optional, Sequence, Tuple, Type

import numpy as np
from torch import Tensor, nn

from ..prelude import ArrayLike
from ..utils import Device
from .block import CNNBody, FCBody, LinearHead, NetworkBlock, ResNetBody
from .init import Initializer, orthogonal
from .policy import CategoricalDist, Policy, PolicyDist
from .prelude import NetFn
from .recurrent import DummyRnn, RnnBlock, RnnState


class ActorCriticNet(nn.Module, ABC):
    """Network with Policy + Value Head
    """

    state_dim: Sequence[int]
    action_dim: int
    recurrent_body: RnnBlock

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


class SeparatedACNet(nn.Module, ABC):
    """Separated Actor Critic network
    """

    def __init__(
        self,
        actor_body: NetworkBlock,
        critic_body: NetworkBlock,
        policy_dist: PolicyDist,
        device: Device = Device(),
        init: Initializer = Initializer(),
    ) -> None:
        super().__init__()
        self.device = device
        self.actor = nn.Sequential(
            actor_body,
            LinearHead(actor_body.output_dim, policy_dist.input_dim, init=init),
        )
        self.critic = nn.Sequential(
            critic_body, LinearHead(critic_body.output_dim, 1, init=init)
        )
        self.policy_dist = policy_dist
        self.recurrent_body = DummyRnn()
        self.to(device.unwrapped)
        self.state_dim = self.actor[0].input_dim
        self.action_dim = self.actor[1].output_dim

    @property
    def is_recurrent(self) -> bool:
        return False

    def policy(
        self,
        states: ArrayLike,
        _rnns: Optional[RnnState] = None,
        _masks: Optional[Tensor] = None,
    ) -> Tuple[Policy, RnnState]:
        s = self.device.tensor(states)
        return self.policy_dist(self.actor(s)), self.recurrent_body.DUMMY_STATE

    def value(
        self,
        states: ArrayLike,
        _rnns: Optional[RnnState] = None,
        _masks: Optional[Tensor] = None,
    ) -> Tensor:
        s = self.device.tensor(states)
        return self.critic(s).squeeze_()

    def forward(
        self,
        states: ArrayLike,
        _rnns: Optional[RnnState] = None,
        _masks: Optional[Tensor] = None,
    ) -> Tuple[Policy, Tensor, RnnState]:
        s = self.device.tensor(states)
        policy = self.policy_dist(self.actor(s))
        value = self.critic(s).squeeze_()
        return policy, value, self.recurrent_body.DUMMY_STATE


class SharedACNet(ActorCriticNet):
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
        self.device = device
        self.body = body
        self.actor_head = actor_head
        self.critic_head = critic_head
        self.policy_dist = policy_dist
        self.recurrent_body = recurrent_body
        self.to(device.unwrapped)
        self.state_dim = self.body.input_dim
        self.action_dim = self.actor_head.output_dim

    def _features(
        self,
        states: ArrayLike,
        rnns: Optional[RnnState] = None,
        masks: Optional[Tensor] = None,
    ) -> Tuple[Tensor, RnnState]:
        res = self.body(self.device.tensor(states))
        if rnns is None:
            rnns = self.recurrent_body.initial_state(res.size(0), self.device)
        res = self.recurrent_body(res, rnns, masks)
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
        return self.critic_head(features).squeeze_()

    def forward(
        self,
        states: ArrayLike,
        rnns: Optional[RnnState] = None,
        masks: Optional[Tensor] = None,
    ) -> Tuple[Policy, Tensor, RnnState]:
        features, rnn_next = self._features(states, rnns, masks)
        policy, value = self.actor_head(features), self.critic_head(features)
        return self.policy_dist(policy), value.squeeze_(), rnn_next


def policy_init(gain: float = 0.01) -> Initializer:
    """Use small value for policy layer to make policy entroy larger
    """
    return Initializer(weight_init=orthogonal(gain))


def _make_ac_shared(
    body: NetworkBlock, policy_dist: PolicyDist, device: Device, rnn: Type[RnnBlock],
) -> SharedACNet:
    rnn_ = rnn(body.output_dim, body.output_dim)
    ac_head = LinearHead(body.output_dim, policy_dist.input_dim, policy_init())
    cr_head = LinearHead(body.output_dim, 1)
    return SharedACNet(
        body, ac_head, cr_head, policy_dist, recurrent_body=rnn_, device=device
    )


def conv_shared(
    policy: Type[PolicyDist] = CategoricalDist,
    cnn_params: Sequence[tuple] = [(8, 4), (4, 2), (3, 1)],
    hidden_channels: Sequence[int] = (32, 64, 32),
    feature_dim: int = 256,
    rnn: Type[RnnBlock] = DummyRnn,
    **kwargs,
) -> NetFn:
    """Convolutuion network used for atari experiments
       in A3C paper(https://arxiv.org/abs/1602.01783)
    """

    def _net(
        state_dim: Tuple[int, int, int], action_dim: int, device: Device
    ) -> SharedACNet:
        body = CNNBody(
            state_dim,
            cnn_params=cnn_params,
            hidden_channels=hidden_channels,
            output_dim=feature_dim,
            **kwargs,
        )
        policy_dist = policy(action_dim, device)
        return _make_ac_shared(body, policy_dist, device, rnn)

    return _net  # type: ignore


def fc_shared(
    policy: Type[PolicyDist] = CategoricalDist, rnn: Type[RnnBlock] = DummyRnn, **kwargs
) -> NetFn:
    """FC body head ActorCritic network
    """

    def _net(state_dim: Sequence[int], action_dim: int, device: Device) -> SharedACNet:
        body = FCBody(state_dim[0], **kwargs)
        policy_dist = policy(action_dim, device)
        return _make_ac_shared(body, policy_dist, device, rnn)

    return _net


def fc_separated(
    actor_units: List[int] = [64, 64],
    critic_units: List[int] = [64, 64],
    init: Initializer = Initializer(),
    policy_type: Type[PolicyDist] = CategoricalDist,
) -> NetFn:
    """SAC network with separated bodys
    """

    def _net(
        state_dim: Sequence[int], action_dim: int, device: Device
    ) -> SeparatedACNet:
        actor_body = FCBody(state_dim[0], units=actor_units, init=init)
        critic_body = FCBody(state_dim[0], units=critic_units, init=init)
        policy = policy_type(action_dim)
        return SeparatedACNet(actor_body, critic_body, policy, device=device, init=init)

    return _net


def impala_conv(
    policy: Type[PolicyDist] = CategoricalDist,
    channels: List[int] = [16, 32, 32],
    rnn: Type[RnnBlock] = DummyRnn,
    **kwargs,
) -> NetFn:
    """Convolutuion network used in IMPALA
    """

    def _net(
        state_dim: Tuple[int, int, int], action_dim: int, device: Device
    ) -> SharedACNet:
        body = ResNetBody(state_dim, channels, **kwargs)
        policy_dist = policy(action_dim, device)
        return _make_ac_shared(body, policy_dist, device, rnn)

    return _net  # type: ignore
