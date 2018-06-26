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
    def input_dim(self) -> int:
        pass

    @abstractmethod
    def output_dim(self) -> int:
        pass

    @abstractmethod
    def predict(self, x: ndarray) -> Tensor:
        pass


class LinearHead(NetworkHead):
    def __init__(
            self,
            output_dim: int,
            body: NetworkBody,
            device: Device = Device(),
            init: Initializer = Initializer()
    ) -> None:
        super(LinearHead, self).__init__()
        self.dev = device
        self.op_dim = output_dim
        self.fc = init(nn.Linear(body.output_dim(), output_dim))
        self.body = body

    def input_dim(self) -> int:
        return self.body.input_dim()

    def output_dim(self) -> int:
        return self.op_dim

    def predict(self, x: ndarray) -> Tensor:
        x = self.device.tensor(x)
        x = self.body(x)
        x = self.fc(x)
        return x
