from typing import Any, Iterable, List, Sequence, Tuple, TypeVar, Union

from torch import Tensor

T = TypeVar("T")
Self = Any
Index = Union[None, int, slice, Tensor, List[Any], Tuple[Any, ...]]
Params = Union[Iterable[Tensor], dict]


class Array(Sequence[T]):
    @property
    def shape(self) -> tuple:
        ...

    def squeeze(self) -> Self:
        ...

    def transpose(self, *args) -> Self:
        ...

    def mean(self, **kwargs) -> Self:
        ...

    def var(self, **kwargs) -> Self:
        ...

    def __rsub__(self, value: Any) -> Self:
        ...

    def __truediv__(self, rvalue: Any) -> Self:
        ...


ArrayLike = Union[Array[Any], List[Any], Tensor]

Action = TypeVar("Action", int, Array)
State = TypeVar("State")


DEFAULT_SAVEFILE_NAME = "rainy-agent.pth"
