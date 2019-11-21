from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from typing import Generic, Iterable, Optional, Tuple, TypeVar
from .init import lstm_bias, Initializer
from ..prelude import Index, Self
from ..utils import Device


class RnnState(ABC):
    @abstractmethod
    def __getitem__(self, x: Index) -> Self:
        pass

    @abstractmethod
    def __setitem__(self, x: Index, value: Self) -> None:
        pass

    @abstractmethod
    def fill_(self, f: float) -> None:
        pass

    @abstractmethod
    def unsqueeze(self) -> Self:
        pass


RS = TypeVar("RS", bound=RnnState)


class RnnBlock(Generic[RS], nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(
        self, x: Tensor, hidden: RS, masks: Optional[Tensor]
    ) -> Tuple[Tensor, RS]:
        pass

    @abstractmethod
    def initial_state(self, batch_size: int, device: Device) -> RS:
        pass


def _apply_mask1(mask: Optional[Tensor], x: Tensor) -> Tensor:
    if mask is None:
        return x.unsqueeze(0)
    else:
        return mask.view(1, -1, 1).mul(x)


def _apply_mask2(
    mask: Optional[Tensor], x1: Tensor, x2: Tensor
) -> Tuple[Tensor, Tensor]:
    if mask is None:
        return x1.unsqueeze(0), x2.unsqueeze(0)
    else:
        m = mask.view(1, -1, 1)
        return m.mul(x1), m.mul(x2)


@torch.jit.script
def _reshape_batch(
    x: Tensor, mask: Optional[Tensor], nsteps: int
) -> Tuple[Tensor, Tensor]:
    x = x.view(nsteps, -1, x.size(-1))
    if mask is None:
        return x, torch.ones_like(x[:, :, 0])
    else:
        return x, mask.view(nsteps, -1)


def _haszero_iter(mask: Tensor, nstep: int) -> Iterable[Tuple[int, int]]:
    has_zeros = (mask[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
    if has_zeros.dim() == 0:
        haszero = [int(has_zeros.item() + 1)]
    else:
        haszero = (has_zeros + 1).tolist()
    return zip([0] + haszero, haszero + [nstep])


class LstmState(RnnState):
    def __init__(self, h: Tensor, c: Tensor, squeeze: bool = True) -> None:
        self.h = h
        self.c = c
        if squeeze:
            self.h.squeeze_(0)
            self.c.squeeze_(0)

    def __getitem__(self, x: Index) -> Self:
        return LstmState(self.h[x], self.c[x])

    def __setitem__(self, x: Index, value: Self) -> None:
        self.h[x] = value.h[x]
        self.c[x] = value.c[x]

    def fill_(self, f: float) -> None:
        self.h.fill_(f)
        self.c.fill_(f)

    def mul_(self, x: Tensor) -> None:
        self.h.mul_(x)
        self.c.mul_(x)

    def unsqueeze(self) -> Self:
        return LstmState(self.h.unsqueeze(0), self.c.unsqueeze(0))


class LstmBlock(RnnBlock[LstmState]):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initializer: Initializer = Initializer(bias_init=lstm_bias()),
        **kwargs
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.lstm = nn.LSTM(input_dim, output_dim, **kwargs)
        initializer(self.lstm)

    def forward(
        self, x: Tensor, hidden: LstmState, mask_: Optional[Tensor] = None
    ) -> Tuple[Tensor, LstmState]:
        in_shape = x.shape
        if in_shape == hidden.h.shape:
            out, (h, c) = self.lstm(
                x.unsqueeze(0), _apply_mask2(mask_, hidden.h, hidden.c)
            )
            return out.squeeze(0), LstmState(h, c)
        # forward Nsteps altogether
        nsteps = in_shape[0] // hidden.h.size(0)
        x, mask = _reshape_batch(x, mask_, nsteps)
        res, h, c = [], hidden.h, hidden.c
        for start, end in _haszero_iter(mask, nsteps):
            m = mask[start].view(1, -1, 1)
            processed, (h, c) = self.lstm(x[start:end], (h * m, c * m))
            res.append(processed)
        return torch.cat(res).view(in_shape), LstmState(h, c)

    def initial_state(self, batch_size: int, device: Device) -> LstmState:
        zeros = device.zeros((batch_size, self.input_dim))
        return LstmState(zeros, zeros, squeeze=False)


class GruState(RnnState):
    def __init__(self, h: Tensor) -> None:
        self.h = h

    def __getitem__(self, x: Index) -> Self:
        return GruState(self.h[x])

    def __setitem__(self, x: Index, value: Self) -> None:
        self.h[x] = value.h[x]

    def fill_(self, f: float) -> None:
        self.h.fill_(f)

    def mul_(self, x: Tensor) -> None:
        self.h.mul_(x)

    def unsqueeze(self) -> Self:
        return GruState(self.h.unsqueeze(0))


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

    def forward(
        self, x: Tensor, hidden: GruState, mask_: Optional[Tensor] = None
    ) -> Tuple[Tensor, GruState]:
        in_shape = x.shape
        if in_shape == hidden.h.shape:
            out, h = self.gru(x.unsqueeze(0), _apply_mask1(mask_, hidden.h))
            return out.squeeze(0), GruState(h.squeeze_(0))  # type: ignore
        # forward Nsteps altogether
        nsteps = in_shape[0] // hidden.h.size(0)
        x, mask = _reshape_batch(x, mask_, nsteps)
        res, h = [], hidden.h
        for start, end in _haszero_iter(mask, nsteps):
            processed, h = self.gru(x[start:end], h * mask[start].view(1, -1, 1))
            res.append(processed)
        return torch.cat(res).view(in_shape), GruState(h.squeeze_(0))

    def initial_state(self, batch_size: int, device: Device) -> GruState:
        return GruState(device.zeros((batch_size, self.input_dim)))


class DummyState(RnnState):
    def __getitem__(self, x: Index) -> Self:
        return self

    def __setitem__(self, x: Index, value: Self) -> None:
        pass

    def fill_(self, f: float) -> None:
        pass

    def unsqueeze(self) -> Self:
        return self


class DummyRnn(RnnBlock[DummyState]):
    DUMMY_STATE = DummyState()

    def __init__(self, *args, **kwargs) -> None:
        nn.Module.__init__(self)

    def forward(
        self,
        x: Tensor,
        hidden: DummyState = DUMMY_STATE,
        mask: Optional[Tensor] = None,
    ) -> Tuple[Tensor, DummyState]:
        return x, hidden

    def initial_state(self, batch_size: int, device: Device) -> DummyState:
        return self.DUMMY_STATE
