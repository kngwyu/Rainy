from typing import Optional, Tuple

import numpy as np
import pytest
import torch
from test_env import DummyEnv

from rainy.net import (
    CNNBody,
    CNNBodyWithoutFC,
    GruBlock,
    LstmBlock,
    actor_critic,
    termination_critic,
)
from rainy.net.init import Initializer, kaiming_normal, kaiming_uniform
from rainy.utils import Device

ACTION_DIM = 10


@pytest.mark.parametrize(
    "net, state_dim, batch_size",
    [
        (actor_critic.fc_shared()((4,), ACTION_DIM, Device()), (4,), 32),
        (
            actor_critic.conv_shared()((4, 84, 84), ACTION_DIM, Device()),
            (4, 84, 84),
            32,
        ),
        (
            actor_critic.conv_shared(rnn=GruBlock)((4, 84, 84), ACTION_DIM, Device()),
            (4, 84, 84),
            32,
        ),
        (
            actor_critic.conv_shared(rnn=LstmBlock)((4, 84, 84), ACTION_DIM, Device()),
            (4, 84, 84),
            32,
        ),
        (
            actor_critic.impala_conv()((4, 84, 84), ACTION_DIM, Device()),
            (4, 84, 84),
            32,
        ),
    ],
)
def test_acnet(
    net: actor_critic.ActorCriticNet, state_dim: tuple, batch_size: int
) -> None:
    assert net.state_dim == state_dim
    assert net.action_dim == ACTION_DIM
    env = DummyEnv()
    states = np.stack(
        [env.step(None)[0].to_array(state_dim) for _ in range(batch_size)]
    )
    policy, values, _ = net(states)
    batch_size = torch.Size([batch_size])
    assert policy.action().shape == batch_size
    assert policy.log_prob().shape == batch_size
    assert policy.entropy().shape == batch_size
    assert values.shape == batch_size


@pytest.mark.parametrize(
    "input_dim, batch_size, params, channels, hidden, init",
    [
        (
            (4, 84, 84),
            32,
            None,
            None,
            (64, 7, 7),
            Initializer(weight_init=kaiming_normal(nonlinearity="relu")),
        ),
        ((4, 84, 84), 64, None, None, (64, 7, 7), None),
        (
            (17, 48, 24),
            64,
            [(8, 1), (4, 1), (3, 1)],
            None,
            (64, 36, 12),
            Initializer(weight_init=kaiming_uniform(nonlinearity="leaky_relu")),
        ),
        ((17, 32, 16), 64, [(8, 1), (4, 1), (3, 1)], None, (64, 20, 4), None),
        (
            (3, 210, 160),
            32,
            [(8, 2, (1, 1)), (6, 2, (1, 1)), (6, 2, (1, 1)), (4, 2)],
            (64, 128, 128, 128),
            (128, 11, 8),
            None,
        ),
    ],
)
def test_convbody(
    input_dim: Tuple[int, int, int],
    batch_size: int,
    params: Optional[list],
    channels: Optional[tuple],
    hidden: Tuple[int, int, int],
    init: Optional[Initializer],
) -> None:
    kwargs = {}
    if params is not None:
        kwargs["cnn_params"] = params
    if init is not None:
        kwargs["init"] = init
    if channels is not None:
        kwargs["hidden_channels"] = channels
    conv = CNNBody(input_dim, **kwargs)
    assert conv.fc.in_features == np.prod(hidden)
    x = torch.ones((batch_size, *input_dim))
    x = conv(x)
    assert x.size(0) == batch_size
    assert x.size(1) == conv.output_dim

    without_fc = CNNBodyWithoutFC(input_dim, **kwargs)
    assert without_fc.output_dim == conv.fc.in_features


@pytest.mark.parametrize("state_dim", [(2, 64, 64), (100,)])
def test_tcnet(state_dim: tuple):
    BATCH_SIZE = 10
    NUM_OPTIONS = 3

    if len(state_dim) > 1:
        net_fn = termination_critic.tc_conv_shared(num_options=NUM_OPTIONS)
    else:
        net_fn = termination_critic.tc_fc_shared(num_options=NUM_OPTIONS)

    net = net_fn(state_dim, 1, Device())
    input1 = torch.randn(BATCH_SIZE, *state_dim)
    input2 = torch.randn(BATCH_SIZE, *state_dim)
    out = net(input1, input2)
    assert tuple(out.beta.dist.logits.shape) == (BATCH_SIZE, NUM_OPTIONS)
    assert tuple(out.p.shape) == (BATCH_SIZE, NUM_OPTIONS)
    assert tuple(out.p_mu.shape) == (BATCH_SIZE, NUM_OPTIONS)
    assert tuple(out.baseline.shape) == (BATCH_SIZE, NUM_OPTIONS)
