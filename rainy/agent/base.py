from abc import ABC, abstractmethod
from numpy import ndarray
import torch
from torch import nn
from typing import Any, Tuple
from ..net import NetworkHead, NetworkBody, Initializer


class Agent(ABC):
    @abstractmethod
    def members_to_save(self) -> Tuple[str, ...]:
        """Here you can specify members you want to save.

    Examples::
        def members_to_save(self):
            return "net", "target"
        """
        pass

    # @abstractmethod
    # def action(self, state: ndarray):
    #     pass

    def save(self, filename: str) -> None:
        save_dict = {}
        for idx, member_str in enumerate(self.members_to_save()):
            value = getattr(self, member_str)
            if isinstance(value, nn.Module):
                save_dict[idx] = value.state_dict()
            else:
                save_dict[idx] = value
        torch.save(save_dict, filename)

    def load(self, filename: str) -> None:
        saved_dict = torch.load(filename)
        for idx, member_str in enumerate(self.members_to_save()):
            saved_item = saved_dict[idx]
            if isinstance(saved_item, nn.Module):
                mod = getattr(self, member_str)
                mod.load_state_dict(saved_item)
            else:
                setattr(self, member_str, saved_item)

