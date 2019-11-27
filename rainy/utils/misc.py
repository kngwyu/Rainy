import torch
from torch import Tensor


@torch.jit.script
def clamp_actions_(t: Tensor, min_t: Tensor, max_t: Tensor) -> Tensor:
    for i in range(t.size(1)):
        t[:, i].clamp_(min_t[i].item(), max_t[i].item())
    return t


@torch.jit.script
def normalize_(t: Tensor, eps: float) -> None:
    mean = t.mean()
    std = t.std()
    t.sub_(mean).div_(std + eps)
