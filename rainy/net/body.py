# network bodies
from abc import ABC, abstractmethod
from functools import reduce
import operator
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Callable, List, Tuple
from .init import Initializer


Activator = Callable[[Tensor], Tensor]


class NetworkBody(nn.Module, ABC):
    @property
    @abstractmethod
    def input_dim(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass


class ConvBody(NetworkBody):
    def __init__(
            self,
            activator: Activator,
            init: Initializer,
            input_dim: Tuple[int, int, int],
            hidden_dim: Tuple[int, int, int],
            fc: nn.Linear,
            *args
    ) -> None:
        super().__init__()
        self.conv = init.make_list(*args)
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.fc = init(fc)
        self.init = init
        self.activator = activator

    @property
    def input_dim(self) -> Tuple[int, ...]:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self.fc.out_features

    @property
    def hidden_dim(self) -> Tuple[int, int, int]:
        return self._hidden_dim

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.conv:
            x = self.activator(conv(x))
        x = x.view(x.size(0), -1)
        x = self.activator(self.fc(x))
        return x


def calc_cnn_offset(params: List[Tuple[int, int]], width: int, height: int) -> Tuple[int, int]:
    for kernel, stride in params:
        width = (width - kernel) // stride + 1
        height = (height - kernel) // stride + 1
    assert width > 0 and height > 0, 'Convolution makes dim < 0!!!'
    return width, height


class DqnConv(ConvBody):
    """Convolutuion Network used in https://www.nature.com/articles/nature14236,
       but parameterized to use in A2C or else.
    """
    def __init__(
            self,
            dim: Tuple[int, int, int],
            kernel_and_strides: List[Tuple[int, int]] = [(8, 4), (4, 2), (3, 1)],
            hidden_channels: Tuple[int, int, int] = (32, 64, 64),
            output_dim: int = 512,
            activator: Activator = F.relu,
            init: Initializer = Initializer(nonlinearity = 'relu')
    ) -> None:
        in_channel, width, height = dim
        hidden1, hidden2, hidden3 = hidden_channels
        conv1 = nn.Conv2d(in_channel, hidden1, *kernel_and_strides[0])
        conv2 = nn.Conv2d(hidden1, hidden2, *kernel_and_strides[1])
        conv3 = nn.Conv2d(hidden2, hidden3, *kernel_and_strides[2])
        hidden = (hidden3, *calc_cnn_offset(kernel_and_strides, width, height))
        fc = nn.Linear(reduce(operator.mul, hidden), output_dim)
        self._output_dim = output_dim
        super().__init__(F.relu, init, dim, hidden, fc, conv1, conv2, conv3)


class FcBody(nn.Module):
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
    def input_dim(self) -> Tuple[int, ...]:
        return (self.dims[0],)

    @property
    def output_dim(self) -> int:
        return self.dims[-1]

