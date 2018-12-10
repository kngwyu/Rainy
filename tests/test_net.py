import numpy as np
import pytest
from rainy.net import actor_critic, DqnConv
from rainy.util import Device
from test_env import DummyEnv
import torch
from typing import Optional, Tuple


ACTION_DIM = 10
BATCH_SIZE = 10


@pytest.mark.parametrize('net, state_dim', [
    (actor_critic.fc((4,), ACTION_DIM, Device()), (4,)),
])
def test_acnet(net: actor_critic.ActorCriticNet, state_dim: Tuple[int, ...]) -> None:
    assert net.state_dim == state_dim
    assert net.action_dim == ACTION_DIM
    env = DummyEnv()
    states = np.stack([env.step(None)[0].to_array(len(state_dim)) for _ in range(BATCH_SIZE)])
    actions, log_prob, entropy, values = map(torch.detach, net(states))
    assert actions.shape == torch.Size([BATCH_SIZE])
    assert log_prob.shape == torch.Size([BATCH_SIZE])
    assert entropy.shape == torch.Size([BATCH_SIZE])
    assert values.shape == torch.Size([BATCH_SIZE])


@pytest.mark.parametrize('input_dim, batch_size, params', [
    ((4, 84, 84), 32, None),
    ((4, 84, 84), 64, None),
    ((17, 48, 24), 64, [(8, 1), (4, 1), (3, 1)]),
])
def test_dqnconv(input_dim: Tuple[int, int, int], batch_size: int, params: Optional[list]) -> None:
    dqn_conv = DqnConv(input_dim, params=params) if params else DqnConv(input_dim)
    x = torch.ones((batch_size, *input_dim))
    for conv in dqn_conv.conv:
        x = conv(x)
    assert x.shape[0] == batch_size
    in_features = x.shape[1] * x.shape[2] * x.shape[3]
    assert in_features == dqn_conv.fc.in_features
