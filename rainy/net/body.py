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
            fc: nn.Linear,
            *args
    ) -> None:
        super().__init__()
        self.conv = init.make_list(*args)
        self._input_dim = input_dim
        self.fc = init(fc)
        self.init = init
        self.activator = activator

    @property
    def input_dim(self) -> Tuple[int, ...]:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self.fc.out_features

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.conv:
            x = self.activator(conv(x))
        x = x.view(x.size(0), -1)
        x = self.activator(self.fc(x))
        return x


def calc_cnn_offset(params: List[Tuple[int, int]], img_size: Tuple[int, int]) -> Tuple[int, int]:
    width, height = img_size
    for kernel, stride in params:
        width = (width - (kernel // stride) + 1) // stride
        height = (height - (kernel // stride) + 1) // stride
    return width, height


class DqnConv(ConvBody):
    """Convolutuion Network used in https://www.nature.com/articles/nature14236,
       but parameterized to use in A2C or else.
    """
    def __init__(
            self,
            dim: Tuple[int, int, int],
            batch_size: int = 32,
            kernel_and_strides: List[Tuple[int, int]] = [(8, 4), (4, 2), (3, 1)],
            hidden_channels: Tuple[int, int] = (64, 64),
            output_dim: int = 512,
            activator: Activator = F.relu,
            init: Initializer = Initializer(nonlinearity = 'relu')
    ) -> None:
        in_channel, width, height = dim
        hidden1, hidden2 = hidden_channels
        conv1 = nn.Conv2d(in_channel, batch_size, *kernel_and_strides[0])
        conv2 = nn.Conv2d(batch_size, hidden1, *kernel_and_strides[1])
        conv3 = nn.Conv2d(hidden1, hidden2, *kernel_and_strides[2])
        width, height = calc_cnn_offset(kernel_and_strides, (width, height))
        assert width != 0 and height != 0
        fc = nn.Linear(width * height * hidden2, output_dim)
        self._output_dim = output_dim
        super().__init__(F.relu, init, dim, fc, conv1, conv2, conv3)


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

