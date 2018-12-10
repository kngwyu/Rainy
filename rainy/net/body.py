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
            fc: nn.Linear,
            init: Initializer,
            *args
    ) -> None:
        super().__init__()
        self.conv = init.make_list(*args)
        self.fc = init(fc)
        self.init = init
        self.activator = activator

    @property
    def input_dim(self) -> Tuple[int, ...]:
        return (self.conv[0].in_channels, self.conv[0].out_channels)

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
    """Convolutuion Network used in https://www.nature.com/articles/nature14236
    """
    def __init__(
            self,
            dim: Tuple[int, int, int],
            batch_size: int = 32,
            params: List[Tuple[int, ...]] = [(8, 4), (4, 2), (3, 1)],
            activator: Activator = F.relu,
            init: Initializer = Initializer()
    ) -> None:
        channel, width, height = dim
        self._output_dim = 512
        conv1 = nn.Conv2d(channel, batch_size, *params[0])
        conv2 = nn.Conv2d(32, 64, *params[1])
        conv3 = nn.Conv2d(64, 64, *params[2])
        width, height = calc_cnn_offset(params, (width, height))
        assert width != 0 and height != 0
        fc = nn.Linear(64 * width * height, self._output_dim)
        super().__init__(F.relu, fc, init, conv1, conv2, conv3)


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

