from functools import partial
from typing import Callable, Iterable, Optional, Union
from torch import nn, Tensor

# function to init Tensor
InitFn = Callable[[Tensor], None]


def uniform(mean: float = 0., var: float = 1.) -> InitFn:
    return partial(nn.init.uniform_, a=mean, b=var)


def orthogonal(gain: float = 1.) -> InitFn:
    return partial(nn.init.orthogonal_, gain=gain)


def kaiming_normal(gain: float = 1.) -> InitFn:
    return partial(nn.init.kaiming_normal_, gain=gain)


def kaiming_uniform(gain: float = 1.) -> InitFn:
    return partial(nn.init.kaiming_uniform_, gain=gain)


def constant(val: float) -> InitFn:
    return partial(nn.init.constant_, val=val)


def zero() -> InitFn:
    return partial(nn.init.constant_, val=0)


class Initializer:
    """Utility Class to initialize weight parameters of NN
    """
    def __init__(
            self,
            nonlinearity: Optional[str] = None,
            weight_init: InitFn = orthogonal(),
            bias_init: InitFn = zero(),
            scale: float = 1.,
    ) -> None:
        """If nonlinearity is specified, use orthogonal
           with calucurated gain by torch.init.calculate_gain.
        """
        self.weight_init = weight_init
        if nonlinearity:
            gain = nn.init.calculate_gain(nonlinearity)
            if 'gain' in self.weight_init.keywords:
                self.weight_init.keywords['gain'] = gain
            else:
                raise ValueError('{} doesn\'t have gain', self.weight_init)
        self.bias_init = bias_init
        self.scale = scale

    def __call__(self, mod: Union[nn.Module, nn.Sequential, Iterable[nn.Module]]) -> nn.Module:
        if hasattr(mod, '__iter__'):
            self.__init_list(mod)
        else:
            self.__init_mod(mod)
        return mod

    def make_list(self, *args) -> nn.ModuleList:
        return nn.ModuleList([self.__init_mod(mod) for mod in args])

    def make_seq(self, *args) -> nn.Sequential:
        return nn.Sequential(*map(lambda mod: self.__init_mod(mod), args))

    def __init_mod(self, mod: nn.Module) -> nn.Module:
        for name, param in mod.named_parameters():
            if 'weight' in name:
                self.weight_init(param)
            if 'bias' in name:
                self.bias_init(param)
        return mod

    def __init_list(self, mods: Iterable[nn.Module]) -> Iterable[nn.Module]:
        for mod in mods:
            self.__init_mod(mod)
        return mods


