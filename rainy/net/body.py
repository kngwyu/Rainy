# network bodies
from abc import ABC, abstractmethod
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Callable, List, Tuple
from .init import Initializer


Activator = Callable[[Tensor], Tensor]


class NetworkBody(nn.Module, ABC):
    @property
    @abstractmethod
    def input_dim(self) -> Tuple[int]:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass


class DqnConv(NetworkBody):
    """Convolutuion Network used in https://www.nature.com/articles/nature14236
    """
    def __init__(
            self,
            in_channels: int,
            batch_size: int = 32,
            activator: Activator = F.relu,
            init: Initializer = Initializer()
    ) -> None:
        super().__init__()
        self._output_dim = 512
        self.activator = activator
        conv1 = nn.Conv2d(in_channels, batch_size, kernel_size=8, stride=4)
        conv2 = nn.Conv2d(batch_size, 64, kernel_size=4, stride=2)
        conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv = init.make_list(conv1, conv2, conv3)
        self.fc = init(nn.Linear(7 * 7 * 64, self.output_dim))
        self.init = init

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.conv:
            x = self.activator(conv(x))
        x = x.view(x.size(0), -1)
        x = self.activator(self.fc(x))
        return x

    @property
    def input_dim(self) -> Tuple[int]:
        return (0,)

    @property
    def output_dim(self) -> int:
        return self._output_dim


class FullyConnected(nn.Module):
    def __init__(
            self,
            input_dim: int,
            units: List[int] = [64, 64],
            activator: Activator = F.relu,
            init: Initializer = Initializer()
    ) -> None:
        super().__init__()
        self._input_dim = input_dim
        dims = [input_dim] + units
        self.layers = init.make_list(*map(lambda i, o: nn.Linear(i, o), *zip(dims[:-1], dims[1:])))
        self.activator = activator
        self.dims = dims

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = self.activator(layer(x))
        return x

    @property
    def input_dim(self) -> Tuple[int]:
        return (self.dims[0],)

    @property
    def output_dim(self) -> int:
        return self.dims[-1]

