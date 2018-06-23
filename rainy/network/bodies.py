# network bodies
from abc import ABC, abstractmethod
from enum import Enum
from torch import nn, Tensor
import torch.nn.functional as F

class NetworkBody(nn.Module, ABC):
    @abstractmethod
    def feature_dim(self) -> int:
        pass


class NatureDQNConv(NetworkBody):
    """Convolutuion Network in https://www.nature.com/articles/nature14236
    """
    def __init__(self, input_channels:int = 4) -> None:
        super(NatureDQNConv, self).__init__()
        self.__feature_dim = 512
        conv1 = nn.Conv2d(input_channels, 32, kernel_size=8, stride=4)
        conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        self.fc4 = nn.Linear(7 * 7 * 64, self.__feature_dim)

    def feature_dim(self) -> int:
        return self.__feature_dim

    def forward(self, x: Tensor) -> Tensor:
        x = F.relu(self.conv1(x))