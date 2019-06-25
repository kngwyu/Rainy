import torch
from torch import Tensor
from typing import Optional


@torch.jit.script
def normalize_(t: Tensor, eps: float) -> None:
    mean = t.mean()
    std = t.std()
    t.sub_(mean).div_(std + eps)


def has_freq_in_interval(turn: int, width: int, freq: Optional[int]) -> bool:
    return freq and turn != 0 and turn // freq != (turn - width) // freq  # type: ignore
