import numpy as np
import pytest
from rainy.net import actor_critic, DqnConv
from rainy.net.init import Initializer, kaiming_normal, kaiming_uniform
from rainy.utils import Device
from test_env import DummyEnv
import torch
from typing import Optional, Tuple


ACTION_DIM = 10


@pytest.mark.parametrize('net, state_dim, batch_size', [
    (actor_critic.fc()((4,), ACTION_DIM, Device()), (4,), 32),
    (actor_critic.ac_conv()((4, 84, 84), ACTION_DIM, Device()), (4, 84, 84), 32),
    (actor_critic.impala_conv()((4, 84, 84), ACTION_DIM, Device()), (4, 84, 84), 32),
])
def test_acnet(net: actor_critic.ActorCriticNet, state_dim: tuple, batch_size: int) -> None:
    assert net.state_dim == state_dim
    assert net.action_dim == ACTION_DIM
    env = DummyEnv()
    states = np.stack([env.step(None)[0].to_array(state_dim) for _ in range(batch_size)])
    policy, values = net(states)
    batch_size = torch.Size([batch_size])
    assert policy.action().shape == batch_size
    assert policy.log_prob().shape == batch_size
    assert policy.entropy().shape == batch_size
    assert values.shape == batch_size


@pytest.mark.parametrize('input_dim, batch_size, params, hidden, init', [
    ((4, 84, 84), 32, None, (64, 7, 7),
     Initializer(weight_init=kaiming_normal(), nonlinearity='relu')),
    ((4, 84, 84), 64, None, (64, 7, 7), None),
    ((17, 48, 24), 64, [(8, 1), (4, 1), (3, 1)], (64, 36, 12),
     Initializer(weight_init=kaiming_uniform(), nonlinearity='leaky_relu')),
    ((17, 32, 16), 64, [(8, 1), (4, 1), (3, 1)], (64, 20, 4), None),
])
def test_dqnconv(
        input_dim: Tuple[int, int, int],
        batch_size: int,
        params: Optional[list],
        hidden: Tuple[int, int, int],
        init: Optional[Initializer],
) -> None:
    kwargs = {}
    if params:
        kwargs['kernel_and_strides'] = params
    if init:
        kwargs['init'] = init
    dqn_conv = DqnConv(input_dim, **kwargs)
    assert dqn_conv.hidden_dim == hidden
    x = torch.ones((batch_size, *input_dim))
    for conv in dqn_conv.conv:
        x = conv.forward(x)
    assert x.shape[0] == batch_size
    in_features = x.shape[1] * x.shape[2] * x.shape[3]
    assert in_features == dqn_conv.fc.in_features
