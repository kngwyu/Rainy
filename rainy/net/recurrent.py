from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from typing import Generic, List, Tuple, TypeVar
from .init import Initializer
from ..prelude import Self
from ..utils import Device


class RnnState(ABC):
    @abstractmethod
    def mul_(self, x: Tensor) -> None:
        pass

    @abstractmethod
    def __getitem__(self, x: Tensor) -> Self:
        pass


RS = TypeVar('RS', bound=RnnState)


class RnnBlock(Generic[RS], nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor, hidden: RS) -> Tuple[Tensor, RS]:
        pass

    @abstractmethod
    def make_batch(self, hiddens: List[RS]) -> RS:
        pass

    @abstractmethod
    def initial_state(self, batch_size: int, device: Device) -> RS:
        pass


class LstmState(RnnState):
    def __init__(self, h: Tensor, c: Tensor) -> None:
        self.h = h
        self.c = c

    def mul_(self, x: Tensor) -> None:
        self.h.mul_(x)
        self.c.mul_(x)

    def __getitem__(self, x: Tensor) -> Self:
        return LstmState(self.h.squeeze()[x].unsqueeze(0), self.c.squeeze()[x].unsqueeze(0))


class LstmBlock(RnnBlock[LstmState]):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            initializer: Initializer = Initializer(),
            **kwargs
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.lstm = nn.LSTM(input_dim, output_dim, **kwargs)
        initializer(self.lstm)

    def make_batch(self, hiddens: List[LstmState]) -> LstmState:
        h = torch.cat([h.h.squeeze_() for h in hiddens])
        c = torch.cat([h.c.squeeze_() for h in hiddens])
        return LstmState(h.unsqueeze_(0), c.unsqueeze_(0))

    def forward(self, x: Tensor, hidden: LstmState) -> Tuple[Tensor, LstmState]:
        if len(x.shape) == 2:
            x.unsqueeze_(0)
        out, next_h = self.lstm(x, (hidden.h, hidden.c))
        return out.squeeze(), LstmState(*next_h)

    def initial_state(self, batch_size: int, device: Device) -> LstmState:
        zeros = device.zeros((1, batch_size, self.input_dim))
        return LstmState(zeros, zeros)


class GruState(RnnState):
    def __init__(self, h: Tensor) -> None:
        self.h = h

    def mul_(self, x: Tensor) -> None:
        self.h.mul_(x)

    def __getitem__(self, x: Tensor) -> Self:
        return GruState(self.h.squeeze()[x].unsqueeze(0))


class GruBlock(RnnBlock[GruState]):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            initializer: Initializer = Initializer(),
            **kwargs
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.gru = nn.GRU(input_dim, output_dim, **kwargs)
        initializer(self.gru)

    def make_batch(self, hiddens: List[GruState]) -> GruState:
        return GruState(torch.cat([h.h.squeeze_() for h in hiddens]).unsqueeze_(0))

    def forward(self, x: Tensor, hidden: GruState) -> Tuple[Tensor, GruState]:
        if len(x.shape) == 2:
            x.unsqueeze_(0)
        out, next_h = self.gru(x, hidden.h)
        return out.squeeze(), GruState(next_h)

    def initial_state(self, batch_size: int, device: Device) -> GruState:
        return GruState(device.zeros((1, batch_size, self.input_dim)))


class DummyState(RnnState):
    def mul_(self, x: Tensor) -> None:
        pass

    def __getitem__(self, x: Tensor) -> Self:
        return self


class DummyRnn(RnnBlock[DummyState]):
    DUMMY_STATE = DummyState()

    def __init__(self, *args, **kwargs) -> None:
        nn.Module.__init__(self)

    def make_batch(self, hiddens: List[DummyState]) -> DummyState:
        return self.DUMMY_STATE

    def forward(self, x: Tensor, hidden: DummyState) -> Tuple[Tensor, DummyState]:
        return x, hidden

    def initial_state(self, batch_size: int, device: Device) -> DummyState:
        return self.DUMMY_STATE
