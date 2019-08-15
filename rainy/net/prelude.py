from torch import nn
from typing import Callable, Tuple
from ..utils.device import Device

NetFn = Callable[[Tuple[int, ...], int, Device], nn.Module]
