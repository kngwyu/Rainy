from torch.optim import Optimizer
from typing import Any, Iterable

class OptimizerGen:
    def __init__(self):
        pass
    def __call__(self, params: Iterable[Any]) -> Optimizer:
        pass

