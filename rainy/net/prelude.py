from torch import nn
from typing import Callable, Sequence
from ..utils.device import Device

NetFn = Callable[[Sequence[int], int, Device], nn.Module]
