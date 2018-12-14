try:
    from typing import GenericMeta, NamedTupleMeta  # type: ignore

    class GenericNamedMeta(NamedTupleMeta, GenericMeta):
        pass
except ImportError:
    from typing import NamedTupleMeta  # type: ignore
    GenericNamedMeta = NamedTupleMeta  # type: ignore

from typing import _alias  # type: ignore
from typing import TypeVar
from numpy import ndarray

T = TypeVar('T')
NdArray = _alias(ndarray, T, inst=False)
