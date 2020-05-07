"""Defines some reusable NN layers, called 'Block'
"""
from abc import ABC
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from .init import Initializer, orthogonal


class NetworkBlock(nn.Module, ABC):
    """Defines a NN block which returns 1-dimension Tensor
    """

    input_dim: Sequence[int]
    output_dim: int


class DummyBlock(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class LinearHead(NetworkBlock):
    """One FC layer
    """

    def __init__(
        self, input_dim: int, output_dim: int, init: Initializer = Initializer()
    ) -> None:
        super().__init__()
        self.fc: nn.Linear = init(nn.Linear(input_dim, output_dim))  # type: ignore
        self.input_dim = (input_dim,)
        self.output_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.fc(x)


class RPFLinearHead(NetworkBlock):
    """FC layer with Randomized Prior function(https://arxiv.org/abs/1806.03335)
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        init: Initializer = Initializer(),
        prior_init: Optional[Initializer] = None,
        prior_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.fc_raw: nn.Linear = init(nn.Linear(input_dim, output_dim))  # type: ignore
        if prior_init is None:
            self.fc_prior: nn.Linear = init(nn.Linear(input_dim, output_dim))
        else:
            self.fc_prior: nn.Linear = prior_init(nn.Linear(input_dim, output_dim))
        self.prior_scale = prior_scale
        self.input_dim = (input_dim,)
        self.output_dim = output_dim

    def forward(self, x: Tensor) -> Tensor:
        with torch.no_grad():
            prior = self.fc_prior(x)
        return self.fc_raw(x).add_(prior.mul_(self.prior_scale))


class CNNBody(NetworkBlock):
    """Multiple CNN layers + FC
    By default, this is the same as CNN used in the DQN paper.
    """

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        cnn_params: Sequence[tuple] = [(8, 4), (4, 2), (3, 1)],
        hidden_channels: Sequence[int] = (32, 64, 64),
        output_dim: int = 512,
        activator: nn.Module = nn.ReLU(inplace=True),
        init: Initializer = Initializer(orthogonal(nonlinearity="relu")),
    ) -> None:
        super().__init__()
        cnns, hidden = make_cnns(input_dim, cnn_params, hidden_channels)
        self.cnns = init.make_list(cnns)
        self.fc = init(nn.Linear(hidden, output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activator = activator

    def forward(self, x: Tensor) -> Tensor:
        for cnn in self.cnns:
            x = self.activator(cnn(x))
        return self.activator(self.fc(x.reshape(x.size(0), -1)))


class CNNBodyWithoutFC(NetworkBlock):
    """Almost the same as CNNBody, but has no FC layer
    """

    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        cnn_params: Sequence[Tuple[int, int]] = [(8, 4), (4, 2), (3, 1)],
        hidden_channels: Sequence[int] = (32, 64, 64),
        activator: nn.Module = nn.ReLU(inplace=True),
        init: Initializer = Initializer(orthogonal(nonlinearity="relu")),
    ) -> None:
        super().__init__()
        cnns, hidden = make_cnns(input_dim, cnn_params, hidden_channels)
        self.cnns = init.make_list(cnns)
        self.input_dim = input_dim
        self.output_dim = hidden
        self.activator = activator

    def forward(self, x: Tensor) -> Tensor:
        for cnn in self.cnns:
            x = self.activator(cnn(x))
        return x


class BatchNormCNN(NetworkBlock):
    def __init__(
        self,
        input_dim: Tuple[int, int, int],
        cnn_params: Sequence[tuple] = [(8, 4), (4, 2), (3, 1)],
        hidden_channels: Sequence[int] = (32, 64, 64),
        output_dim: int = 512,
        activator: nn.Module = nn.ReLU(inplace=True),
        init: Initializer = Initializer(orthogonal(nonlinearity="relu")),
    ) -> None:
        super().__init__()
        cnns, hidden = make_cnns(input_dim, cnn_params, hidden_channels)
        self.cnns = init.make_list(cnns)
        self.batch_norms = init.make_list(map(nn.BatchNorm2d, hidden_channels))
        self.fc = init(nn.Linear(hidden, output_dim))
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.activator = activator

    def forward(self, x: Tensor) -> Tensor:
        for cnn, bn in zip(self.cnns, self.batch_norms):
            x = self.activator(bn(cnn(x)))
        return self.activator(self.fc(x.reshape(x.size(0), -1)))


class ResBlock(nn.Sequential):
    def __init__(
        self, channel: int, stride: int = 1, use_batch_norm: bool = True,
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
        init: Initializer = Initializer(orthogonal(nonlinearity="relu")),
    ) -> None:
        def layer(in_channel: int, out_channel: int, maxpool: tuple) -> nn.Sequential:
            return nn.Sequential(
                ResBlock._conv3x3(in_channel, out_channel),
                ResBlock._batch_norm(use_batch_norm, out_channel),
                nn.MaxPool2d(*maxpool),
                ResBlock(out_channel, use_batch_norm=use_batch_norm),
                ResBlock(out_channel, use_batch_norm=use_batch_norm),
            )

        super().__init__()

        self.input_dim = input_dim
        self.output_dim = fc_out
        self.res_blocks = init.make_list(
            [layer(*t) for t in zip([input_dim[0]] + channels, channels, maxpools)]
        )
        self.relu = nn.ReLU(inplace=True)
        conved = calc_cnn_hidden(maxpools, *input_dim[1:])
        fc_in = np.prod((channels[-1], *conved))
        self.fc = nn.Linear(fc_in, fc_out)

    def forward(self, x: Tensor) -> Tensor:
        for block in self.res_blocks:
            x = block(x)
        x = self.relu(x)
        return self.relu(self.fc(x.reshape(x.size(0), -1)))


class FCBody(NetworkBlock):
    def __init__(
        self,
        input_dim: int,
        units: List[int] = [64, 64],
        activator: nn.Module = nn.ReLU(inplace=True),
        init: Initializer = Initializer(),
    ) -> None:
        super().__init__()
        self.input_dim = (input_dim,)
        self.output_dim = units[-1]
        dims = [input_dim] + units
        self.layers = init.make_list(
            map(lambda i, o: nn.Linear(i, o), *zip(dims[:-1], dims[1:]))
        )
        self.activator = activator

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = self.activator(layer(x))
        return x


class BatchNormFC(NetworkBlock):
    def __init__(
        self,
        input_dim: int,
        units: List[int] = [64, 64],
        activator: nn.Module = nn.ReLU(inplace=True),
        init: Initializer = Initializer(),
    ) -> None:
        super().__init__()
        self.input_dim = (input_dim,)
        self.output_dim = units[-1]
        dims = [input_dim] + units
        self.fcs = init.make_list(
            map(lambda i, o: nn.Linear(i, o), *zip(dims[:-1], dims[1:]))
        )
        self.batch_norms = init.make_list(map(nn.BatchNorm1d, units))
        self.activator = activator

    def forward(self, x: Tensor) -> Tensor:
        for fc, bn in zip(self.fcs, self.batch_norms):
            x = self.activator(bn(fc(x)))
        return x


def make_cnns(
    input_dim: Tuple[int, int, int],
    params: Sequence[tuple],
    hidden_channels: Sequence[int],
) -> Tuple[List[nn.Conv2d], int]:
    """Make a list of CNNs from lists of parameters.
    """
    channel, height, width = input_dim  # input is CHW
    hiddens = list(hidden_channels)
    res = []
    for ic, oc, param in zip([channel] + hiddens, hiddens, params):
        res.append(nn.Conv2d(ic, oc, *param))
    hidden = (hidden_channels[-1], *calc_cnn_hidden(params, height, width))
    return res, np.prod(hidden)


def calc_cnn_hidden(
    params: Sequence[tuple], height: int, width: int
) -> Tuple[int, int]:
    """Calcurate hidden dim of a CNN.
       See https://pytorch.org/docs/stable/nn.html#torch.nn.Conv2d for detail.
    """
    for param in params:
        kernel, stride, pad = param if len(param) > 2 else (*param, 0)
        if isinstance(pad, int):
            h_pad, w_pad = pad, pad
        else:
            h_pad, w_pad = pad
        height = (height - kernel + 2 * h_pad) // stride + 1
        width = (width - kernel + 2 * w_pad) // stride + 1
    assert width > 0 and height > 0, "Convolution makes dim < 0!!!"
    return height, width
