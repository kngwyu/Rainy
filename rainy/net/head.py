# network heads
from abc import ABC, abstractmethod
from numpy import ndarray
import torch
from torch import nn, Tensor
import torch.nn.functional as F
from .body import NetworkBody
from .init import Initializer
from ..lib import Device


class NetworkHead(nn.Module, ABC):
    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass

    @abstractmethod
    def predict(self, x: ndarray) -> Tensor:
        pass

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        data = torch.load(filename)
        self.load_state_dict(data)

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
        self.__output_dim = output_dim
        self.fc = init(nn.Linear(body.output_dim, output_dim))
        self.body = body

    @property
    def input_dim(self) -> int:
        return self.body.input_dim

    @property
    def output_dim(self) -> int:
        return self.__output_dim

    def predict(self, x: ndarray) -> Tensor:
        x = self.device.tensor(x)
        x = self.body(x)
        x = self.fc(x)
        return x
