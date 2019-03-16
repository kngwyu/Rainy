"""Defines some reusable NN layers, named as 'Block'
"""
from abc import ABC, abstractmethod
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Callable, List, Sequence, Tuple
from .init import Initializer
from ..utils.misc import iter_prod

Activator = Callable[[Tensor], Tensor]


class NetworkBlock(nn.Module, ABC):
    """Defines a NN block which returns 1-dimension Tensor
    """
    @property
    @abstractmethod
    def input_dim(self) -> Tuple[int, ...]:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass


class DummyBlock(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class LinearHead(NetworkBlock):
    """One FC layer
    """
    def __init__(self, input_dim: int, output_dim: int, init: Initializer = Initializer()) -> None:
        super().__init__()
        self._input_dim = input_dim
        self._output_dim = output_dim
        self.fc = init(nn.Linear(input_dim, output_dim))

    @property
    def input_dim(self) -> Tuple[int, ...]:
        return (self._input_dim, )

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class ConvBody(NetworkBlock):
    """Multiple CNN layers + FC
    """
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


class DqnConv(ConvBody):
    """Convolutuion Network used in https://www.nature.com/articles/nature14236,
       but is parameterized for other usages.
    """
    def __init__(
            self,
            input_dim: Tuple[int, int, int],
            kernel_and_strides: List[Tuple[int, int]] = [(8, 4), (4, 2), (3, 1)],
            hidden_channels: Tuple[int, int, int] = (32, 64, 64),
            output_dim: int = 512,
            activator: Activator = F.relu,
            init: Initializer = Initializer(nonlinearity = 'relu')
    ) -> None:
        in_channel, width, height = input_dim
        hidden1, hidden2, hidden3 = hidden_channels
        conv1 = nn.Conv2d(in_channel, hidden1, *kernel_and_strides[0])
        conv2 = nn.Conv2d(hidden1, hidden2, *kernel_and_strides[1])
        conv3 = nn.Conv2d(hidden2, hidden3, *kernel_and_strides[2])
        hidden = (hidden3, *calc_cnn_hidden(kernel_and_strides, width, height))
        fc = nn.Linear(iter_prod(hidden), output_dim)
        self._output_dim = output_dim
        super().__init__(F.relu, init, input_dim, hidden, fc, conv1, conv2, conv3)


class ResBlock(nn.Sequential):
    def __init__(
            self,
            channel: int,
            stride: int = 1,
            use_batch_norm: bool = True,
    ) -> None:
        super().__init__(
            nn.ReLU(inplace=True),
            self._conv3x3(channel, channel, stride),
            self._batch_norm(use_batch_norm, channel),
            nn.ReLU(inplace=True),
            self._conv3x3(channel, channel, stride),
            self._batch_norm(use_batch_norm, channel),
        )

    @staticmethod
    def _batch_norm(use_batch_norm: bool, channel: int) -> nn.Module:
        if use_batch_norm:
            return nn.BatchNorm2d(channel)
        else:
            return DummyBlock()

    @staticmethod
    def _conv3x3(in_channel: int, out_channel: int, stride: int = 1) -> nn.Conv2d:
        return nn.Conv2d(
            in_channel,
            out_channel,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False,
        )

    def forward(self, x: Tensor) -> Tensor:
        residual = x
        out = super().forward(x)
        return out + residual


class ResNetBody(NetworkBlock):
    """Convolutuion Network used in IMPALA
    """
    def __init__(
            self,
            input_dim: Tuple[int, int, int],
            channels: List[int] = [16, 32, 32],
            maxpools: List[tuple] = [(3, 2, 1)] * 3,
            use_batch_norm: bool = True,
            fc_out: int = 256,
            init: Initializer = Initializer(nonlinearity = 'relu'),
    ) -> None:
        def layer(in_channel: int, out_channel: int, maxpool: tuple) -> nn.Sequential:
            return nn.Sequential(
                ResBlock._conv3x3(in_channel, out_channel),
                ResBlock._batch_norm(use_batch_norm, out_channel),
                nn.MaxPool2d(*maxpool),
                ResBlock(out_channel, use_batch_norm=use_batch_norm),
                ResBlock(out_channel, use_batch_norm=use_batch_norm)
            )

        super().__init__()
        self._input_dim = input_dim
        self.res_blocks = init.make_list(*[
            layer(*t) for t in zip([input_dim[0]] + channels, channels, maxpools)
        ])
        self.relu = nn.ReLU(inplace=True)
        conved = calc_cnn_hidden(maxpools, *input_dim[1:])
        fc_in = iter_prod((channels[-1], *conved))
        self.fc = nn.Linear(fc_in, fc_out)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.res_blocks:
            x = block(x)
        x = self.relu(x)
        x = self.fc(x.view(x.size(0), -1))
        return self.relu(x)

    @property
    def input_dim(self) -> Tuple[int, ...]:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self.fc.out_features


class FcBody(NetworkBlock):
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


def calc_cnn_hidden(params: Sequence[tuple], width: int, height: int) -> Tuple[int, int]:
    """Calcurate hidden dim of a CNN.
       See https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d for detail.
    """
    for param in params:
        kernel, stride, padding = param if len(param) > 2 else (*param, 0)
        width = (width - kernel + 2 * padding) // stride + 1
        height = (height - kernel + 2 * padding) // stride + 1
    assert width > 0 and height > 0, 'Convolution makes dim < 0!!!'
    return width, height
