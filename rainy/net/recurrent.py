from abc import ABC, abstractmethod
import torch
from torch import nn, Tensor
from typing import Generic, Iterable, Optional, Sequence, Tuple, TypeVar
from .init import constant, forget_bias, Initializer
from ..prelude import Self
from ..utils import Device


class RnnState(ABC):
    @abstractmethod
    def __getitem__(self, x: Sequence[int]) -> Self:
        pass

    @abstractmethod
    def fill_(self, f: float) -> None:
        pass

    @abstractmethod
    def mul_(self, x: Tensor) -> None:
        pass


RS = TypeVar('RS', bound=RnnState)


class RnnBlock(Generic[RS], nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    @abstractmethod
    def forward(self, x: Tensor, hidden: RS, masks: Optional[Tensor]) -> Tuple[Tensor, RS]:
        pass

    @abstractmethod
    def initial_state(self, batch_size: int, device: Device) -> RS:
        pass


def _apply_mask(x: RnnState, mask: Optional[Tensor]) -> None:
    if mask is not None:
        x.mul_(mask.unsqueeze(1))


def _reshape_batch(x: Tensor, mask: Optional[Tensor], nsteps: int) -> Tuple[Tensor, Tensor]:
    x = x.view(nsteps, -1, x.size(-1))
    if mask is None:
        return x, torch.ones_like(x[:, :, 0])
    else:
        return x, mask.view(nsteps, -1)


def _haszero_iter(mask: Tensor, nstep: int) -> Iterable[Tuple[int, int]]:
    has_zeros = (mask[1:] == 0.0).any(dim=-1).nonzero().squeeze().cpu()
    if has_zeros.dim() == 0:
        haszero = [has_zeros.item() + 1]
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

    def __getitem__(self, x: Sequence[int]) -> Self:
        return LstmState(self.h[x], self.c[x])

    def fill_(self, f: float) -> None:
        self.h.fill_(f)
        self.c.fill_(f)

    def mul_(self, x: Tensor) -> None:
        self.h.mul_(x)
        self.c.mul_(x)


class LstmBlock(RnnBlock[LstmState]):
    def __init__(
            self,
            input_dim: int,
            output_dim: int,
            initializer: Initializer = Initializer(bias_init = forget_bias(1.0)),
            **kwargs
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.lstm = nn.LSTM(input_dim, output_dim, **kwargs)
        initializer(self.lstm)

    def forward(
            self,
            x: Tensor,
            hidden: LstmState,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, LstmState]:
        in_shape = x.shape
        if in_shape == hidden.h.shape:
            _apply_mask(hidden, mask)
            out, (h, c) = self.lstm(x.unsqueeze(0), (hidden.h.unsqueeze(0), hidden.c.unsqueeze(0)))
            return out.squeeze(0), LstmState(h, c)
        # forward Nsteps altogether
        nsteps = in_shape[0] // hidden.h.size(0)
        x, mask = _reshape_batch(x, mask, nsteps)
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

    def __getitem__(self, x: Sequence[int]) -> Self:
        return GruState(self.h[x])

    def fill_(self, f: float) -> None:
        self.h.fill_(f)

    def mul_(self, x: Tensor) -> None:
        self.h.mul_(x)


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

    def forward(
            self,
            x: Tensor,
            hidden: GruState,
            mask: Optional[Tensor] = None
    ) -> Tuple[Tensor, GruState]:
        in_shape = x.shape
        if in_shape == hidden.h.shape:
            _apply_mask(hidden, mask)
            out, h = self.gru(x.unsqueeze(0), hidden.h.unsqueeze(0))
            return out.squeeze(0), GruState(h.squeeze_(0))
        # forward Nsteps altogether
        nsteps = in_shape[0] // hidden.h.size(0)
        x, mask = _reshape_batch(x, mask, nsteps)
        res, h = [], hidden.h
        for start, end in _haszero_iter(mask, nsteps):
            processed, h = self.gru(x[start:end], h * mask[start].view(1, -1, 1))
            res.append(processed)
        return torch.cat(res).view(in_shape), GruState(h.squeeze_(0))

    def initial_state(self, batch_size: int, device: Device) -> GruState:
        return GruState(device.zeros((batch_size, self.input_dim)))


class DummyState(RnnState):
    def __getitem__(self, x: Sequence[int]) -> Self:
        return self

    def fill_(self, f: float) -> None:
        pass

    def mul_(self, x: Tensor) -> None:
        pass


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
