import numpy as np
import pytest
from rainy.net import actor_critic
from rainy.util import Device
from test_env import DummyEnv
import torch
from typing import Tuple


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
