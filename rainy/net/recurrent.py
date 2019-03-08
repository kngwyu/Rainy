from abc import abstractmethod
import torch
from torch import nn, Tensor
from typing import Any, Generic, List, NamedTuple, Optional, Tuple, TypeVar
from .init import Initializer
from ..prelude import Array

RnnState = TypeVar('RnnState')


class RnnBlock(Generic[RnnState], nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor, hidden: Optional[RnnState]) -> Tuple[Tensor, RnnState]:
        pass

    @abstractmethod
    def make_batch(self, hiddens: List[RnnState], masks: List[Array[float]]) -> RnnState:
        pass


class LstmState(NamedTuple):
    hidden: Tensor
    cell: Tensor


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

    def make_batch(self, hiddens: List[LstmState], masks: List[Array[float]]) -> RnnState:
        pass

    def forward(self, x: Tensor, hidden: Optional[LstmState]) -> Tuple[Tensor, LstmState]:
        pass


class GruState(NamedTuple):
    hidden: Tensor


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

    def make_batch(self, hiddens: List[GruState], masks: List[Array[float]]) -> RnnState:
        return torch.cat([h.hidden for h in hiddens])

    def forward(self, x: Tensor, hidden: Optional[GruState]) -> Tuple[Tensor, GruState]:
        if len(x.shape) == 2:
            x.unsqueeze_(dim=0)
            res = self.gru(x, hidden)
            return res[0].squeeze_(), GruState(res[1])
        else:
            raise NotImplementedError('')


class DummyState:
    def __getitem__(self, idx: Any) -> Any:
        return self


class DummyRnn(RnnBlock[DummyState]):
    def __init__(self, *args, **kwargs) -> None:
        nn.Module.__init__(self)

    def make_batch(self, hiddens: List[DummyState], masks: List[Array[float]]) -> RnnState:
        return DummyState()

    def forward(self, x: Tensor, hidden: Optional[DummyState]) -> Tuple[Tensor, DummyState]:
        return x, hidden
