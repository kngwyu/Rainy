from typing import Callable

import pytest
import torch

from rainy.net import DilatedRnnBlock, GruBlock, LstmBlock, RnnBlock
from rainy.utils import Device


def dilated_gru(input_dim: int, output_dim: int) -> DilatedRnnBlock:
    return DilatedRnnBlock(GruBlock(input_dim, output_dim), 4)


@pytest.mark.parametrize(
    "rnn_gen", [GruBlock, LstmBlock, dilated_gru],
)
def test_rnn(rnn_gen: Callable[[int, int], RnnBlock]) -> None:
    TIME_STEP = 10
    BATCH_SIZE = 5
    INPUT_DIM = 20
    OUTPUT_DIM = 3
    rnn = rnn_gen(INPUT_DIM, OUTPUT_DIM)
    device = Device()
    rnn.to(device.unwrapped)
    hidden = rnn.initial_state(BATCH_SIZE, device)
    cached_inputs = []

    for i in range(TIME_STEP):
        inputs = torch.randn(BATCH_SIZE, INPUT_DIM, device=device.unwrapped)
        cached_inputs.append(inputs.detach())
        out, hidden = rnn(inputs, hidden)
        assert tuple(out.shape) == (BATCH_SIZE, OUTPUT_DIM)
    batch_inputs = torch.cat(cached_inputs)
    hidden = rnn.initial_state(BATCH_SIZE, device)
    out, _ = rnn(batch_inputs, hidden)
    assert tuple(out.shape) == (TIME_STEP * BATCH_SIZE, OUTPUT_DIM)
