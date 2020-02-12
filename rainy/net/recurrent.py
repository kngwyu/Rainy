from abc import ABC, abstractmethod
from typing import Generic, Iterable, Optional, Tuple, TypeVar

import torch
from torch import Tensor, nn

from ..prelude import Index, Self
from ..utils import Device
from .init import Initializer, lstm_bias


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

    @abstractmethod
    def size(self, index: int) -> int:
        pass


RS = TypeVar("RS", bound=RnnState)


class RnnBlock(Generic[RS], nn.Module):
    def __init__(self, input_dim: int, output_dim: int) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

    def forward(
        self, x: Tensor, hidden: RS, masks: Optional[Tensor] = None
    ) -> Tuple[Tensor, RS]:
        x_size0 = x.size(0)
        batch_size = hidden.size(0)
        if x_size0 == batch_size:
            return self.forward_1step(x, hidden, masks)
        else:
            nsteps = x_size0 // batch_size
            inputs, masks = _reshape_batch(x, masks, nsteps)
            output, hidden = self.forward_nsteps(inputs, hidden, masks, nsteps)
            return output.view(x_size0, self.output_dim), hidden

    @abstractmethod
    def forward_1step(
        self, x: Tensor, hidden: RS, masks: Optional[Tensor]
    ) -> Tuple[Tensor, RS]:
        pass

    @abstractmethod
    def forward_nsteps(
        self, x: Tensor, hidden: RS, masks: Optional[Tensor], nsteps: int,
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

    def size(self, index: int) -> int:
        return self.h.size(index)


class LstmBlock(RnnBlock[LstmState]):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initializer: Initializer = Initializer(bias_init=lstm_bias()),
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.lstm = nn.LSTM(input_dim, output_dim, **kwargs)
        initializer(self.lstm)

    def forward_1step(
        self, x: Tensor, hidden: RS, masks: Optional[Tensor]
    ) -> Tuple[Tensor, RS]:
        out, (h, c) = self.lstm(x.unsqueeze(0), _apply_mask2(masks, hidden.h, hidden.c))
        return out.squeeze(0), LstmState(h, c)

    def forward_nsteps(
        self, x: Tensor, hidden: RS, masks: Optional[Tensor], nsteps: int,
    ) -> Tuple[Tensor, RS]:
        res, h, c = [], hidden.h, hidden.c
        for start, end in _haszero_iter(masks, nsteps):
            m = masks[start].view(1, -1, 1)
            processed, (h, c) = self.lstm(x[start:end], (h * m, c * m))
            res.append(processed)
        return torch.cat(res), LstmState(h, c)

    def initial_state(self, batch_size: int, device: Device) -> LstmState:
        zeros = device.zeros((batch_size, self.output_dim))
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

    def size(self, index: int) -> int:
        return self.h.size(index)


class GruBlock(RnnBlock[GruState]):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        initializer: Initializer = Initializer(),
        **kwargs,
    ) -> None:
        super().__init__(input_dim, output_dim)
        self.gru = nn.GRU(input_dim, output_dim, **kwargs)
        initializer(self.gru)

    def forward_1step(
        self, x: Tensor, hidden: GruState, masks: Optional[Tensor]
    ) -> Tuple[Tensor, GruState]:
        out, h = self.gru(x.unsqueeze(0), _apply_mask1(masks, hidden.h))
        return out.squeeze(0), GruState(h.squeeze_(0))

    def forward_nsteps(
        self, x: Tensor, hidden: GruState, masks: Optional[Tensor], nsteps: int,
    ) -> Tuple[Tensor, GruState]:
        res, h = [], hidden.h
        for start, end in _haszero_iter(masks, nsteps):
            processed, h = self.gru(x[start:end], h * masks[start].view(1, -1, 1))
            res.append(processed)
        return torch.cat(res), GruState(h.squeeze_(0))

    def initial_state(self, batch_size: int, device: Device) -> GruState:
        return GruState(device.zeros((batch_size, self.output_dim)))


class DummyState(RnnState):
    def __getitem__(self, x: Index) -> Self:
        return self

    def __setitem__(self, x: Index, value: Self) -> None:
        pass

    def fill_(self, f: float) -> None:
        pass

    def unsqueeze(self) -> Self:
        return self

    def size(self, _index: int) -> int:
        return 0


class DummyRnn(RnnBlock[DummyState]):
    DUMMY_STATE = DummyState()

    def __init__(self, *args, **kwargs) -> None:
        nn.Module.__init__(self)

    def forward_1step(
        self, x: Tensor, hidden: DummyState, masks: Optional[Tensor]
    ) -> Tuple[Tensor, DummyState]:
        return x, hidden

    def forward_nsteps(
        self, x: Tensor, hidden: DummyState, masks: Optional[Tensor], nsteps: int,
    ) -> Tuple[Tensor, DummyState]:
        return x, hidden

    def forward(
        self, x: Tensor, hidden: DummyState, masks: Optional[Tensor] = None
    ) -> Tuple[Tensor, DummyState]:
        return x, hidden

    def initial_state(self, batch_size: int, device: Device) -> DummyState:
        return self.DUMMY_STATE
