import torch

from ..prelude import Self


class SoftUpdate(torch.nn.Module):
    @torch.no_grad()
    def soft_update(self, other: Self, coef: float) -> None:
        for s_param, o_param in zip(self.parameters(), other.parameters()):
            s_param.copy_(s_param * (1.0 - coef) + o_param * coef)
