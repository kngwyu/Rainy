from numpy import ndarray
from torch import nn, Tensor
from typing import Callable, List, Optional, Tuple, Union
from .block import DqnConv, FcBody, ResNetBody, LinearHead, NetworkBlock
from .init import Initializer, orthogonal
from .policy import CategoricalHead, Policy, PolicyHead
from .recurrent import DummyRnn, RnnBlock, RnnState
from ..prelude import NetFn
from ..utils import Device
from ..utils.misc import iter_prod


class ActorCriticNet(nn.Module):
    """A network with common body, value head and policy head.
    Basically it's same as the one used in A3C paper.
    """
    def __init__(
            self,
            body: NetworkBlock,
            actor_head: NetworkBlock,  # policy
            critic_head: NetworkBlock,  # value
            policy_head: PolicyHead,
            recurrent_body: RnnBlock = DummyRnn(),
            device: Device = Device(),
    ) -> None:
        assert body.output_dim == iter_prod(actor_head.input_dim), \
            'body output and action_head input must have a same dimention'
        assert body.output_dim == iter_prod(critic_head.input_dim), \
            'body output and action_head input must have a same dimention'
        super(ActorCriticNet, self).__init__()
        self.body = body
        self.device = device
        self.body = body
        self.actor_head = actor_head
        self.critic_head = critic_head
        self.policy_head = policy_head
        self.recurrent_body = recurrent_body
        self.to(device.unwrapped)

    @property
    def state_dim(self) -> Tuple[int, ...]:
        return self.body.input_dim

    @property
    def action_dim(self) -> int:
        return self.actor_head.output_dim

    @property
    def is_recurrent(self) -> bool:
        return not isinstance(self.recurrent_body, DummyRnn)

    def _features(
            self,
            states: Union[ndarray, Tensor],
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
            states: Union[ndarray, Tensor],
            rnns: Optional[RnnState] = None,
            masks: Optional[Tensor] = None,
    ) -> Tuple[Policy, RnnState]:
        features, rnns_ = self._features(states, rnns, masks)
        return self.policy_head(self.actor_head(features)), rnns_

    def value(
            self,
            states: Union[ndarray, Tensor],
            rnns: Optional[RnnState] = None,
            masks: Optional[Tensor] = None,
    ) -> Tensor:
        features = self._features(states, rnns, masks)[0]
        return self.critic_head(features).squeeze()

    def forward(
            self,
            states: Union[ndarray, Tensor],
            rnns: Optional[RnnState] = None,
            masks: Optional[Tensor] = None,
    ) -> Tuple[Policy, Tensor, RnnState]:
        features, rnns = self._features(states, rnns, masks)
        policy, value = self.actor_head(features), self.critic_head(features)
        return self.policy_head(policy), value.squeeze(), rnns


def policy_init() -> Initializer:
    """Use small value for policy layer to make policy entroy larger
    """
    return Initializer(weight_init=orthogonal(0.01))


def _make_ac_net(
        body: NetworkBlock,
        policy_head: PolicyHead,
        device: Device,
        rnn: Callable[[int, int], RnnBlock],
) -> ActorCriticNet:
    rnn_ = rnn(body.output_dim, body.output_dim)
    ac_head = LinearHead(body.output_dim, policy_head.input_dim, policy_init())
    cr_head = LinearHead(body.output_dim, 1)
    return ActorCriticNet(body, ac_head, cr_head, policy_head, recurrent_body=rnn_, device=device)


def ac_conv(
        policy: Callable[[int, Device], PolicyHead] = CategoricalHead,
        hidden_channels: Tuple[int, int, int] = (32, 64, 32),
        output_dim: int = 256,
        rnn: Callable[[int, int], RnnBlock] = DummyRnn,
        **kwargs
) -> NetFn:
    """Convolutuion network used for atari experiments
       in A3C paper(https://arxiv.org/abs/1602.01783)
    """
    def _net(state_dim: Tuple[int, int, int], action_dim: int, device: Device) -> ActorCriticNet:
        body = DqnConv(state_dim, hidden_channels=hidden_channels, output_dim=output_dim, **kwargs)
        policy_head = policy(action_dim, device)
        return _make_ac_net(body, policy_head, device, rnn)
    return _net  # type: ignore


def fc_shared(
        policy: Callable[[int, Device], PolicyHead] = CategoricalHead,
        rnn: Callable[[int, int], RnnBlock] = DummyRnn,
        **kwargs
) -> NetFn:
    """FC body head ActorCritic network
    """
    def _net(state_dim: Tuple[int, ...], action_dim: int, device: Device) -> ActorCriticNet:
        body = FcBody(state_dim[0], **kwargs)
        policy_head = policy(action_dim, device)
        return _make_ac_net(body, policy_head, device, rnn)
    return _net


def impala_conv(
        policy: Callable[[int, Device], PolicyHead] = CategoricalHead,
        channels: List[int] = [16, 32, 32],
        rnn: Callable[[int, int], RnnBlock] = DummyRnn,
        **kwargs
) -> NetFn:
    """Convolutuion network used in IMPALA
    """
    def _net(state_dim: Tuple[int, int, int], action_dim: int, device: Device) -> ActorCriticNet:
        body = ResNetBody(state_dim, channels, **kwargs)
        policy_head = policy(action_dim, device)
        return _make_ac_net(body, policy_head, device, rnn)
    return _net  # type: ignore
