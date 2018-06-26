from abc import ABC, abstractmethod
from numpy import ndarray
import torch
from torch import nn
from typing import Any, Tuple
from ..net import NetworkHead, NetworkBody, Initializer


class Agent(ABC):
    @abstractmethod
    def members_to_save(self) -> Tuple:
        """Here you can specify members you want to save.
        **Note** You can save only mutable types.
        You cannot save immutabale types, e.g. int, string.

    Examples::
        def members_to_save(self):
            return self.net, self.target
        """
        pass

    @abstractmethod
    def select_action(self, x: ndarray):
        pass

    def save(self, filename: str) -> None:
        save_dict = {}
        for idx, member in enumerate(self.members_to_save()):
            if isinstance(member, nn.Module):
                save_dict[idx] = member.state_dict()
            else:
                save_dict[idx] = member
        torch.save(save_dict, filename)

    def load(self, filename: str) -> None:
        saved_dict = torch.load(filename)
        for idx, member in enumerate(self.members_to_save()):
            saved_item = saved_dict[idx]
            if isinstance(member, nn.Module):
                member.load_state_dict(saved_item)
            else:
                member = saved_item


from ..net import *
class TestAgent(Agent):
    def __init__(self):
        body = NatureDqnBody(4)
        self.head = LinearHead(4, body)
        self.number = 4
        self.name = "test_agent"

    def save_members(self) -> Tuple:
        return self.head, self.number, self.name

    def select_action(self, x):
        return 5


