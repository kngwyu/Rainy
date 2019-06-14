import torch
from torch import Tensor


@torch.jit.script
def normalize_(t: Tensor, eps: float) -> None:
    mean = t.mean()
    std = t.std()
    t.sub_(mean).div_(std + eps)
