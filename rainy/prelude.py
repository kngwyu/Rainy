from numpy import ndarray
from torch import nn, Tensor
from typing import Callable, Iterable, Tuple, TypeVar, Union
from .utils.device import Device

Action = TypeVar('Action', bound=int)
State = TypeVar('State')

NetFn = Callable[[Tuple[int, ...], int, Device], nn.Module]
Params = Iterable[Union[Tensor, dict]]


try:
    from typing import GenericMeta, NamedTupleMeta  # type: ignore

    class GenericNamedMeta(NamedTupleMeta, GenericMeta):
        pass
except ImportError:
    from typing import NamedTupleMeta  # type: ignore
    GenericNamedMeta = NamedTupleMeta  # type: ignore

T = TypeVar('T')
try:
    from typing import _alias  # type: ignore
    Array = _alias(ndarray, T, inst=False)
except ImportError:
    from typing import Sequence

    class Array(Sequence[T], extra=ndarray):  # type: ignore
        __slots__ = ()

