from abc import ABC, abstractmethod
import torch
from torch import LongTensor, Tensor
from torch.optim import Optimizer
from typing import Optional
from ..net.value import DiscreteQFunction
from ..prelude import Array


class Cooler(ABC):
    @abstractmethod
    def __call__(self) -> float:
        pass

    def lr_decay(self, optim: Optimizer) -> None:
        for param_group in optim.param_groups:
            if "lr" in param_group:
                param_group["lr"] = self.__call__()


class LinearCooler(Cooler):
    """decrease epsilon linearly, from initial to minimal, via `max_step` steps
    """

    def __init__(self, initial: float, minimal: float, max_step: int) -> None:
        if initial < minimal:
            raise ValueError(
                f"[LinearCooler.__init__] the minimal value({minimal})"
                " is bigger than the initial value {initial}"
            )
        self.base = initial - minimal
        self.minimal = minimal
        self.max_step = max_step
        self.left = max_step

    def __call__(self) -> float:
        self.left = max(0, self.left - 1)
        return (float(self.left) / self.max_step) * self.base + self.minimal


class DummyCooler(Cooler):
    """Do nothing
    """

    def __init__(self, initial: float, *args) -> None:
        self.initial = initial

    def __call__(self) -> float:
        return self.initial


class Explorer(ABC):
    def select_action(self, state: Array, qfunc: DiscreteQFunction) -> LongTensor:
        return self.select_from_value(qfunc.q_value(state).detach())

    @abstractmethod
    def select_from_value(self, value: Tensor) -> LongTensor:
        pass

    @abstractmethod
    def add_noise(self, action: Tensor) -> Tensor:
        pass


class Greedy(Explorer):
    """deterministic greedy policy
    """

    def select_from_value(self, value: Tensor) -> LongTensor:
        return value.argmax(-1)  # type: ignore

    def add_noise(self, action: Tensor) -> Tensor:
        return action


class EpsGreedy(Explorer):
    """ε-greedy policy
    """

    def __init__(self, epsilon: float, cooler: Cooler) -> None:
        self.epsilon = epsilon
        self.cooler = cooler

    def select_from_value(self, value: Tensor) -> LongTensor:
        old_eps = self.epsilon
        self.epsilon = self.cooler()
        action_dim = value.size(-1)
        greedy = value.argmax(-1).cpu()
        random = torch.randint_like(greedy, 0, action_dim)
        random_pos = torch.empty(greedy.shape).uniform_() < old_eps
        return torch.where(random_pos, random, greedy)

    def add_noise(self, action: Tensor) -> Tensor:
        raise NotImplementedError("We can't use EpsGreedy with continuous action")


class GaussianNoise(Explorer):
    def __init__(
        self, std: Cooler = DummyCooler(0.1), clip: Optional[float] = None
    ) -> None:
        self.std = std
        self.clip = clip

    def add_noise(self, action: Tensor) -> Tensor:
        noise = torch.randn_like(action).mul_(self.std())
        if self.clip is not None:
            noise.clamp_(-self.clip, self.clip)
        return noise + action

    def select_from_value(self, value: Tensor) -> LongTensor:
        raise NotImplementedError()
