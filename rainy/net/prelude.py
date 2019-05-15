from torch import nn, Tensor
from typing import Callable, Iterable, Tuple, Union
from ..utils.device import Device

NetFn = Callable[[Tuple[int, ...], int, Device], nn.Module]
Params = Iterable[Union[Tensor, dict]]
