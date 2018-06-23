from functools import partial
from typing import Callable
from torch import nn, Tensor

InitFn = Callable[[Tensor], None]

def uniform(mean: float = 0, var: float = 1) -> InitFn:
    return partial(nn.init.uniform_, a=mean, b=var)

def orthiganl(gain: float = 1) -> InitFn:
    return partial(nn.init.orthogonal_, gain=gain)

def zero() -> InitFn:
    return partial(nn.init.constant_, val=0)

def constant(val: float) -> InitFn:
    return partial(nn.init.constant_, val=val)

class Initializer:
    """Utility Class to initialize weight parameters of NN
    """
    def __init__(self, weight_init: InitFn = uniform(), bias_init: InitFn = zero(), scale: float = 1) -> None:
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.scale = scale

    def __call__(self, mod: nn.Module) -> None:
        self.weight_init(mod.weight.data)
        self.bias_init(mod.bias.data)
        mod.weight.data.mul_(self.scale)





