# network heads
from abc import ABC, abstractmethod
from torch import nn, Tensor
from .init import Initializer


class NetworkHead(nn.Module, ABC):
    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass


class LinearHead(NetworkHead):
    def __init__(self, input_dim: int, output_dim: int, init: Initializer = Initializer()) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.fc = init(nn.Linear(input_dim, output_dim))

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class GruHead(NetworkHead):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            num_layers: int = 1,
            init: Initializer = Initializer(),
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.rnn = init(nn.GRU(input_dim, output_dim, num_layers=num_layers))

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Tensor, hidden: Tensor, masks: Tensor):
        pass
