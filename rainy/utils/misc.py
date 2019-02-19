from functools import reduce
import operator
from typing import Any, Iterable


def iter_prod(it: Iterable[Any]) -> Any:
    return reduce(operator.mul, it)
