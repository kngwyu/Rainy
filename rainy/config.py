from torch.optim import Optimizer
from typing import Any, Iterable
from .lib.device import Device
from .net.value_net import ValueNet
from .policy.explore import Explorer

class Config:
    def __init__(self) -> None:
        self.gpu_limits = None
        self.optimizer_gen = None
        self.value_net_gen = None

    def gen_value_net(self, state_dim: int, action_dim: int) -> ValueNet:
        device = self.gen_device()
        return self.value_net_gen(state_dim, action_dim, device)

    def get_explorer(self, valuenet: ValueNet):
        pass
    
    def gen_policy_net(self):
        pass

    def gen_device(self) -> Device:
        return Device(gpu_limits=self.gpu_limits)

    def gen_optimizer(self, params: Iterable[Any]) -> Optimizer:
        return self.optimizer_gen(params)


