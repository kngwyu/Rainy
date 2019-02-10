try:
    from typing import GenericMeta, NamedTupleMeta  # type: ignore

    class GenericNamedMeta(NamedTupleMeta, GenericMeta):
        pass
except ImportError:
    from typing import NamedTupleMeta  # type: ignore
    GenericNamedMeta = NamedTupleMeta  # type: ignore

from numpy import ndarray
from typing import TypeVar
T = TypeVar('T')
try:
    from typing import _alias  # type: ignore
    Array = _alias(ndarray, T, inst=False)
except ImportError:
    from typing import Sequence

    class Array(Sequence[T], extra=ndarray):  # type: ignore
        __slots__ = ()
