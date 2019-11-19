from abc import ABC, abstractmethod
import torch


class HasStateDict(ABC):
    @abstractmethod
    def state_dict(self) -> dict:
        pass

    def load_state_dict(self, d: dict) -> None:
        for key, value in d.items():
            setattr(self, key, value)


class TensorStateDict(HasStateDict):
    def state_dict(self) -> dict:
        all_members = self.__dict__.copy()
        for key in all_members.keys():
            value = all_members[key]
            if isinstance(value, torch.Tensor):
                all_members[key] = value.cpu()
        return all_members

    def load_state_dict(self, d: dict) -> None:
        for key, value in d.items():
            if isinstance(value, torch.Tensor):
                device = getattr(self, key).device
                value = value.to(device)
            setattr(self, key, value)
