from numpy import ndarray
from torch import nn, Tensor
from torch.distributions import Categorical, Distribution
from typing import NamedTuple, Tuple, Union
from .body import DqnConv, FcBody, NetworkBody
from .head import LinearHead, NetworkHead
from ..util import Device


class ActorCriticNet(nn.Module):
    def __init__(
            self,
            body: NetworkBody,
            actor_head: NetworkHead,  # policy
            critic_head: NetworkHead,  # value
            device: Device = Device(),
    ) -> None:
        assert body.output_dim == actor_head.input_dim, \
            'body output and action_head input must have a same dimention'
        assert body.output_dim == critic_head.input_dim, \
            'body output and action_head input must have a same dimention'
        super(ActorCriticNet, self).__init__()
        self.body = body
        self.actor_head = actor_head
        self.critic_head = critic_head
        self.device = device
        self.to(self.device())

    @property
    def state_dim(self) -> Tuple[int, ...]:
        return self.body.input_dim

    @property
    def action_dim(self) -> int:
        return self.actor_head.output_dim

    def best_action(self) -> int:
        raise NotImplementedError()


class AcOutput(NamedTuple):
    policy: Distribution
    value: Tensor


class SoftmaxActorCriticNet(ActorCriticNet):
    def forward(self, states: Union[ndarray, Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        features = self.body(self.device.tensor(states))
        action_prob = self.actor_head(features)
        value = self.critic_head(features)  # [batch_size, 1]
        policy = Categorical(logits=action_prob)  # [batch_size, action_dim]
        return AcOutput(distrib, value)

    def best_action(self, states: Union[ndarray, Tensor]) -> int:
        features = self.body(self.device.tensor(states))
        action_probs = self.actor_head(features).detach()
        dist = Categorical(logits=action_probs)
        return dist._param.argmax()


def ac_conv(state_dim: Tuple[int, int, int], action_dim: int, device: Device) -> ActorCriticNet:
    body = DqnConv(state_dim)
    ac_head = LinearHead(body.output_dim, action_dim)
    cr_head = LinearHead(body.output_dim, 1)
    return SoftmaxActorCriticNet(body, ac_head, cr_head, device=device)


def fc(state_dim: Tuple[int, ...], action_dim: int, device: Device = Device()) -> ActorCriticNet:
    body = FcBody(state_dim[0])
    ac_head = LinearHead(body.output_dim, action_dim)
    cr_head = LinearHead(body.output_dim, 1)
    return SoftmaxActorCriticNet(body, ac_head, cr_head, device=device)

