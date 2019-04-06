from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from typing import Generic, List, Tuple, TypeVar
from .init import constant, Initializer
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
        return LstmState(self.h[x], self.c[x])


class LstmBlock(RnnBlock[LstmState]):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            initializer: Initializer = Initializer(bias_init = constant(1.0)),
            **kwargs
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.lstm = nn.LSTM(input_dim, output_dim, **kwargs)
        initializer(self.lstm)

    def make_batch(self, hiddens: List[LstmState]) -> LstmState:
        h = torch.cat([h.h for h in hiddens])
        c = torch.cat([h.c for h in hiddens])
        return LstmState(h, c)

    def forward(self, x: Tensor, hidden: LstmState) -> Tuple[Tensor, LstmState]:
        out, (h, c) = self.lstm(x.unsqueeze(0), (hidden.h.unsqueeze(0), hidden.c.unsqueeze(0)))
        return out.squeeze(0), LstmState(h.squeeze(0), c.squeeze(0))

    def initial_state(self, batch_size: int, device: Device) -> LstmState:
        zeros = device.zeros((batch_size, self.input_dim))
        return LstmState(zeros, zeros)


class GruState(RnnState):
    def __init__(self, h: Tensor) -> None:
        self.h = h

    def mul_(self, x: Tensor) -> None:
        self.h.mul_(x)

    def __getitem__(self, x: Tensor) -> Self:
        return GruState(self.h[x])


class GruBlock(RnnBlock[GruState]):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            initializer: Initializer = Initializer(bias_init = constant(-1.0)),
            **kwargs
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.gru = nn.GRU(input_dim, output_dim, **kwargs)
        initializer(self.gru)

    def make_batch(self, hiddens: List[GruState]) -> GruState:
        return GruState(torch.cat([h.h for h in hiddens]))

    def forward(self, x: Tensor, hidden: GruState) -> Tuple[Tensor, GruState]:
        out, h = self.gru(x.unsqueeze(0), hidden.h.unsqueeze(0))
        return out.squeeze(0), GruState(h.squeeze(0))

    def initial_state(self, batch_size: int, device: Device) -> GruState:
        return GruState(device.zeros((batch_size, self.input_dim)))


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
