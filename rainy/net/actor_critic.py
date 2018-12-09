import numpy as np
from numpy import ndarray
from torch import nn, Tensor
from torch.distributions import Categorical
import torch.nn.functional as F
from typing import Tuple, Type, Union
from .body import DqnConv, FcBody, NetworkBody
from .head import LinearHead, NetworkHead
from .init import Initializer
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


class DiscreteActorCriticNet(ActorCriticNet):
    def forward(self, states: Union[ndarray, Tensor]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
        features = self.body(self.device.tensor(states))
        action_probs = self.actor_head(features)
        values = self.critic_head(features)  # [batch_size, 1]
        dist = Categorical(logits=action_probs)  # [batch_size, action_dim]
        actions = dist.sample()  # [batch_size]
        log_prob = dist.log_prob(actions)  # [batch_size]
        entropy = dist.entropy()  # [batch_size]
        return actions, log_prob, entropy, values.squeeze()


def fc(state_dim: Tuple[int, ...], action_dim: int, device: Device = Device()) -> ActorCriticNet:
    body = FcBody(state_dim[0])
    ac_head = LinearHead(body.output_dim, action_dim)
    cr_head = LinearHead(body.output_dim, 1)
    return DiscreteActorCriticNet(body, ac_head, cr_head, device=device)

