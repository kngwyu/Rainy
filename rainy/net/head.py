# network heads
from abc import ABC, abstractmethod
from numpy import ndarray
from torch import nn, Tensor
import torch.nn.functional as F
from .body import NetworkBody
from .init import Initializer
from ..util import Device


class NetworkHead(nn.Module, ABC):
    @abstractmethod
    def device(self) -> Device:
        pass

    @abstractmethod
    def input_dim(self) -> int:
        pass

    @abstractmethod
    def output_dim(self) -> int:
        pass

    @abstractmethod
    def predict(self, x: Tensor) -> Tensor:
        pass


class LinearHead(NetworkHead):
    def __init__(
            self,
            output_dim: int,
            body: NetworkBody,
            device: Device = Device(),
            ini: Initializer = Initializer()
    ) -> None:
        super(LinearHead, self).__init__()
        self.dev = device
        self.op_dim = output_dim
        self.fc = ini.init(nn.Linear(body.output_dim(), output_dim))
        self.body = body

    def device(self) -> Device:
        self.dev

    def input_dim(self) -> int:
        self.body.input_dim()

    def output_dim(self) -> int:
        self.op_dim

    def predict(self, x: Tensor) -> Tensor:
        x = self.body(x)
        x = self.fc(x)
        return x
