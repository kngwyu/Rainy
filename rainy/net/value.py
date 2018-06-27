from abc import ABC, abstractmethod
from functools import partial
from numpy import ndarray
import torch
from torch import nn
from .body import NetworkBody, NatureDqnConv
from .head import NetworkHead, LinearHead
from .init import Initializer
from ..lib import Device

class ValueNet(nn.Module, ABC):
    """State -> [Value..]
    """
    @property
    def state_dim(self) -> int:
        self.body.input_dim

    @property
    def action_dim(self) -> int:
        self.head.output_dim

    @abstractmethod
    def get_action_value(self, state: ndarray) -> ndarray:
        pass

    def save(self, filename):
        torch.save(self.state_dict(), filename)

    def load(self, filename):
        data = torch.load(filename)
        self.load_state_dict(data)


class DQN(ValueNet):
    def __init__(
            self,
            body: NetworkBody,
            head: NetworkHead,
            device: Device = Device(),
            init: Initializer = Initializer(),
    ) -> None:
        self.head = head
        self.body = body

    

