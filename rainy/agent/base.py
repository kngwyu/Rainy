from abc import ABC, abstractmethod
from numpy import ndarray
import torch
from typing import Iterable
from ..net import NetworkHead, NetworkBody


class Agent(ABC):
    @abstractmethod
    def network(self) -> NetworkHead:
        pass

    @abstractmethod
    def select_action(self, x: ndarray):
        pass

    def save(self, filename: str) -> None:
        net = self.network()
        torch.save(net.state_dict(), filename)

    def load(self, filename: str) -> None:
        state_dict = torch.load(filename)
        net = self.network()
        net.load_state_dict(state_dict)
