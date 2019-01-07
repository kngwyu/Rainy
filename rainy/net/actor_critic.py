import copy
from numpy import ndarray
from torch import nn, Tensor
from typing import Callable, Tuple, Union
from .body import DqnConv, FcBody, NetworkBody
from .head import LinearHead, NetworkHead
from .init import Initializer, orthogonal
from .policy import categorical, Policy
from ..util import Device


class ActorCriticNet(nn.Module):
    """A network with common body, value head and policy head.
    Basically it's same as the one used in A3C paper.
    """
    def __init__(
            self,
            body: NetworkBody,
            actor_head: NetworkHead,  # policy
            critic_head: NetworkHead,  # value
            policy: Callable[[Tensor], Policy] = categorical,
            device: Device = Device(),
    ) -> None:
        assert body.output_dim == actor_head.input_dim, \
            'body output and action_head input must have a same dimention'
        assert body.output_dim == critic_head.input_dim, \
            'body output and action_head input must have a same dimention'
        super(ActorCriticNet, self).__init__()
        self.body = body
        self.device = device
        self.body = body
        self.actor_head = actor_head
        self.critic_head = critic_head
        self.policy = policy
        self.to(self.device.unwrapped)

    @property
    def state_dim(self) -> Tuple[int, ...]:
        return self.body.input_dim

    @property
    def action_dim(self) -> int:
        return self.actor_head.output_dim

    def policy(self, states: Union[ndarray, Tensor]) -> Policy:
        features = self.body(self.device.tensor(states))
        return self.policy(self.actor_head(features))

    def value(self, states: Union[ndarray, Tensor]) -> Tensor:
        features = self.body(self.device.tensor(states))
        return self.critic_head(features).squeeze()

    def forward(self, states: Union[ndarray, Tensor]) -> Tuple[Policy, Tensor]:
        features = self.body(self.device.tensor(states))
        policy, value = self.actor_head(features), self.critic_head(features)
        return self.policy(policy), value.squeeze()


class RndActorCriticNet(ActorCriticNet):
    """Actor Critic Network with internal reward pridiction head.
    It's used in https://arxiv.org/abs/1810.12894,
    but might be used with other internal reward algorithms.
    """
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(self, *args, **kwargs)
        self.internal_critic_head = copy.deepcopy(self.critic_head)

    def internal_value(self, states: Union[ndarray, Tensor]) -> Tensor:
        features = self.body(self.device.tensor(states))
        return self.internal_critic_head(features).squeeze()


def ac_conv(
        state_dim: Tuple[int, int, int],
        action_dim: int,
        device: Device,
        policy: Callable[[Tensor], Policy] = categorical,
) -> ActorCriticNet:
    """Convolutuion network used for atari experiments
       in A3C paper(https://arxiv.org/abs/1602.01783)
    """
    body = DqnConv(state_dim, hidden_channels=(32, 64, 32), output_dim=256)
    ac_head = LinearHead(body.output_dim, action_dim, Initializer(weight_init=orthogonal(0.01)))
    cr_head = LinearHead(body.output_dim, 1)
    return ActorCriticNet(body, ac_head, cr_head, device=device, policy=policy)


def fc(state_dim: Tuple[int, ...],
       action_dim: int,
       device: Device = Device(),
       policy: Callable[[Tensor], Policy] = categorical) -> ActorCriticNet:
    """FC body head ActorCritic network
    """
    body = FcBody(state_dim[0])
    ac_head = LinearHead(body.output_dim, action_dim, Initializer(weight_init=orthogonal(0.01)))
    cr_head = LinearHead(body.output_dim, 1)
    return ActorCriticNet(body, ac_head, cr_head, device=device, policy=policy)
