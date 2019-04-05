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
    def mul_(self, x: Tensor) -> None:
        pass

    def __getitem__(self, x: Tensor) -> Self:
        pass


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
        pass

    def forward(self, x: Tensor, hidden: LstmState) -> Tuple[Tensor, LstmState]:
        pass

    def initial_state(self, batch_size: int, device: Device) -> LstmState:
        pass


class GruState(RnnState):
    def __init__(self, t: Tensor) -> None:
        self.t = t

    def mul_(self, x: Tensor) -> None:
        self.t.mul_(x)

    def __getitem__(self, x: Tensor) -> Self:
        return GruState(self.t[x].unsqueeze(1))


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
        return GruState(torch.cat([h.t.squeeze_() for h in hiddens], dim=0))

    def forward(self, x: Tensor, hidden: GruState) -> Tuple[Tensor, GruState]:
        if len(x.shape) == 2:
            x.unsqueeze_(dim=0)
        res = self.gru(x, hidden.t)
        return res[0].squeeze(), GruState(res[1])

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
