from functools import partial
from typing import Callable, List
from torch import nn, Tensor

# function to init Tensor
InitFn = Callable[[Tensor], None]


def uniform(mean: float = 0, var: float = 1) -> InitFn:
    return partial(nn.init.uniform_, a=mean, b=var)


def orthogonal(gain: float = 1) -> InitFn:
    return partial(nn.init.orthogonal_, gain=gain)


def constant(val: float) -> InitFn:
    return partial(nn.init.constant_, val=val)


def zero() -> InitFn:
    return partial(nn.init.constant_, val=0)


class Initializer:
    """Utility Class to initialize weight parameters of NN
    """
    def __init__(
            self,
            weight_init: InitFn = uniform(),
            bias_init: InitFn = zero(),
            scale: float = 1
    ) -> None:
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.scale = scale

    def init(self, mod: nn.Module) -> nn.Module:
        self.weight_init(mod.weight.data)
        self.bias_init(mod.bias.data)
        mod.weight.data.mul_(self.scale)
        return mod

    def init_list(self, mods: nn.ModuleList) -> nn.ModuleList:
        for mod in mods:
            self.init(mod)
        return mods

    def make_mod_list(self, mods: List[nn.Module]) -> nn.ModuleList:
        for mod in mods:
            self.init(mod)
        return nn.ModuleList(mods)


