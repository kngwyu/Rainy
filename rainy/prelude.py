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


class Array(Sequence[T]):
    @property
    def shape(self) -> tuple:
        ...

    def squeeze(self) -> Any:
        ...

    def transpose(self, *args) -> Any:
        ...

    def __rsub__(self, value: Any) -> Any:
        ...

    def __truediv__(self, rvalue: Any) -> Any:
        ...


Action = TypeVar('Action', int, Array)
State = TypeVar('State')
NetFn = Callable[[Tuple[int, ...], int, Device], nn.Module]
Params = Iterable[Union[Tensor, dict]]
