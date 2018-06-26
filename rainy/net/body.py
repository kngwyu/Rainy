# network bodies
from abc import ABC, abstractmethod
from torch import nn, Tensor
import torch.nn.functional as F
from typing import Callable
from .init import Initializer


class NetworkBody(nn.Module, ABC):
    @abstractmethod
    def input_dim(self) -> int:
        pass

    @abstractmethod
    def output_dim(self) -> int:
        pass


BodyFn = Callable[[], NetworkBody]


class NatureDqnBody(NetworkBody):
    """Convolutuion Network used in https://www.nature.com/articles/nature14236
    """
    def __init__(
            self,
            input_dim: int,
            init: Initializer = Initializer()
    ) -> None:
        super(NatureDqnBody, self).__init__()
        self.ip_dim = input_dim
        self.op_dim = 512
        conv1 = nn.Conv2d(input_dim, 32, kernel_size=8, stride=4)
        conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)
        self.conv = init.make_list(conv1, conv2, conv3)
        self.fc = init(nn.Linear(7 * 7 * 64, self.op_dim))

    def forward(self, x: Tensor) -> Tensor:
        for conv in self.conv:
            x = F.relu(conv(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc(x))
        return x

    def input_dim(self) -> int:
        return self.ip_dim

    def output_dim(self) -> int:
        return self.op_dim
