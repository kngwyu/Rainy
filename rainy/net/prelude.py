from typing import Callable, Sequence

from torch import nn

from ..utils.device import Device

NetFn = Callable[[Sequence[int], int, Device], nn.Module]
