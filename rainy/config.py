from torch.optim import Optimizer
from typing import List, Optional

class Config:
    def __init__(self) -> None:
        self.optimizer_fn = None
        self.network_fn = None
        self.gpu_limits = None

    def set_gpu_limits(self, gpu_limits: List[int]):
        self.gpu_limits = gpu_limits

    def set_optimizer_fn(self):
        pass

    def get_optimizer(self) -> Optimizer:
        pass

    def get_network(self):
        pass

class ConfigInner:
    def __init__(self, config: Config)-> None:
        pass

