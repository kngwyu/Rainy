# network heads
from abs import ABC, abstractmethod
from numpy import ndarray
from torch import nn, Tensor
import torch.nn.functional as F

from .body import NetworkBody
from .init import Initializer

class NetworkHead(nn.Module, ABC):
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
            ini: Initializer = Initializer()
    ) -> None:
        super(LinearHead, self).__init__()
        self.fc = ini.init(nn.Linear(body.output_dim(), output_dim))
        self.body = body

    def predict(self, x: Tensor) -> Tensor:
        x = self.body(x)
        x = self.fc(x)
        return x
