from functools import partial
from typing import Callable, Iterable, Optional

import numpy as np
import torch
from torch import Tensor, nn

InitFn = Callable[[Tensor], Tensor]


def uniform(min: float = -1.0, max: float = 1.0) -> InitFn:
    return partial(nn.init.uniform_, a=min, b=max)


def normal(mean: float = 0.0, std: float = 1.0) -> InitFn:
    return partial(nn.init.normal_, mean=mean, std=std)


def orthogonal(gain: float = 1.0, nonlinearity: Optional[str] = None) -> InitFn:
    if nonlinearity is not None:
        gain = nn.init.calculate_gain(nonlinearity)
    return partial(nn.init.orthogonal_, gain=gain)


def kaiming_normal(**kwargs) -> InitFn:
    return partial(nn.init.kaiming_normal_, **kwargs)


def kaiming_uniform(**kwargs) -> InitFn:
    return partial(nn.init.kaiming_uniform_, **kwargs)


def xavier_uniform(**kwargs) -> InitFn:
    return partial(nn.init.xavier_uniform_, **kwargs)


def fanin_uniform() -> InitFn:
    def _fanin_uniform(w: Tensor) -> Tensor:
        if w.dim() <= 2:
            fan_in = w.size(0)
        else:
            fan_in = np.prod(w.shape[1:])
        bound = 1.0 / np.sqrt(fan_in)
        return nn.init.uniform_(w, -bound, bound)

    return _fanin_uniform


def constant(val: float) -> InitFn:
    return partial(nn.init.constant_, val=val)


def zero() -> InitFn:
    return partial(nn.init.constant_, val=0)


def lstm_bias(forget: float = 1.0, other: float = 0.0) -> InitFn:
    """Set forget bias and others separately.
    """

    @torch.no_grad()
    def __set_bias(t: Tensor) -> Tensor:
        i = t.size(0) // 4
        t.fill_(other)
        return t[i : 2 * i].fill_(forget)

    return __set_bias


class Initializer:
    """Utility Class to initialize weight parameters of NN
    """

    def __init__(
        self,
        weight_init: InitFn = orthogonal(),
        bias_init: InitFn = zero(),
        scale: float = 1.0,
    ) -> None:
        """If nonlinearity is specified, use orthogonal
           with calucurated gain by torch.init.calculate_gain.
        """
        self.weight_init = weight_init
        self.bias_init = bias_init
        self.scale = scale

    def __call__(self, mod: nn.Module) -> nn.Module:
        return self.__init_dispatch(mod)

    def make_list(self, mods: Iterable[nn.Module]) -> nn.ModuleList:
        return nn.ModuleList([self.__init_dispatch(mod) for mod in mods])

    def __init_dispatch(self, mod: nn.Module) -> nn.Module:
        if isinstance(mod, nn.Sequential) or isinstance(mod, nn.ModuleList):
            for child in mod.children():
                self.__init_dispatch(child)
        else:
            self.__init_mod(mod)
        return mod

    def __init_mod(self, mod: nn.Module) -> nn.Module:
        if "BatchNorm" in mod.__class__.__name__:
            return self.__init_batch_norm(mod)
        for name, param in mod.named_parameters():
            if "weight" in name:
                self.weight_init(param)
            elif "bias" in name:
                self.bias_init(param)
        return mod

    def __init_batch_norm(self, mod: nn.BatchNorm2d) -> nn.Module:
        mod.weight.data.fill_(1.0)
        mod.bias.data.zero_()
        return mod
