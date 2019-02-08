from torch import nn
from torch.optim import Optimizer


class KfacOptimizer(Optimizer):
    def __init__(
            self,
            net: nn.Module,
            eps: float,
            avg_weight: float = 1.0,
            sua: bool = False,
    ) -> None:
        self.eps = eps
        self.avg_weight = avg_weight
        self.sua = sua
        for mod in net.modules():
            mod_class = type(mod).__name__
            if mod_class in ['Linear', 'Conv2d']:
                params = mod.bias
