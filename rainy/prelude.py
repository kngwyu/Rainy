from torch import nn, Tensor
from typing import Any, Callable, Iterable, Sequence, Tuple, TypeVar, Union
from .utils.device import Device


try:
    from typing import GenericMeta, NamedTupleMeta  # type: ignore

    class GenericNamedMeta(NamedTupleMeta, GenericMeta):
        pass
except ImportError:
    from typing import NamedTupleMeta  # type: ignore
    GenericNamedMeta = NamedTupleMeta  # type: ignore

T = TypeVar('T')
Self = Any


class Array(Sequence[T]):
    @property
    def shape(self) -> tuple:
        ...

    def squeeze(self) -> Self:
        ...

    def transpose(self, *args) -> Self:
        ...

    def __rsub__(self, value: Any) -> Self:
        ...

    def __truediv__(self, rvalue: Any) -> Self:
        ...


Action = TypeVar('Action', int, Array)
State = TypeVar('State')
NetFn = Callable[[Tuple[int, ...], int, Device], nn.Module]
Params = Iterable[Union[Tensor, dict]]
