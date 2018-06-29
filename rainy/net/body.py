# network bodies
from abc import ABC, abstractmethod
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Callable
from .init import Initializer


class NetworkBody(nn.Module, ABC):
    @property
    @abstractmethod
    def input_dim(self) -> int:
        pass

    @property
    @abstractmethod
    def output_dim(self) -> int:
        pass


Activator = Callable[[Tensor], Tensor]


class NatureDqnConv(NetworkBody):
    """Convolutuion Network used in https://www.nature.com/articles/nature14236
    """
    def __init__(
            self,
            input_dim: int,
            activator: Activator = F.relu,
            init: Initializer = Initializer()
    ) -> None:
        super(NatureDqnConv, self).__init__()
        self._input_dim = input_dim
        self._output_dim = 512
        self.activator = activator
        conv1 = nn.Conv2d(input_dim, 32, kernel_size=8, stride=4)
        conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv = init.make_list(conv1, conv2, conv3)
        self.fc = init(nn.Linear(7 * 7 * 64, self.output_dim))

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.conv:
            x = self.activator(conv(x))
        x = x.view(x.size(0), -1)
        x = self.activator(self.fc(x))
        return x

    @property
    def input_dim(self) -> int:
        return self._input_dim

    @property
    def output_dim(self) -> int:
        return self._output_dim
