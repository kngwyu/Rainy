"""Dilated RNN implementation: WIP & Currently unused
"""
from typing import Optional, Tuple

import torch
from torch import Tensor

from ..utils import Device
from .recurrent import RS, RnnBlock, _reshape_batch


class DilatedRnnBlock(RnnBlock[RS]):
    def __init__(self, block: RnnBlock, rate: int) -> None:
        super().__init__(block.input_dim, block.output_dim)
        self.block = block
        self.rate = rate

    def forward_1step(
        self, x: Tensor, hidden: RS, masks: Optional[Tensor]
    ) -> Tuple[Tensor, RS]:
        dilated_inputs = self._prepare_inputs(x.unsqueeze(0), 1).squeeze_()
        return self.block.forward_1step(dilated_inputs, hidden, masks)

    def forward_nsteps(
        self, x: Tensor, hidden: RS, masks: Optional[Tensor], nsteps: int,
    ) -> Tuple[Tensor, RS]:
        dilated_inputs = self._prepare_inputs(x, nsteps).squeeze_()
        dilated_masks = self._prepare_masks(masks, nsteps)
        return self.block.forward_nsteps(dilated_inputs, hidden, dilated_masks, nsteps)

    def forward(
        self, x: Tensor, hidden: RS, masks: Optional[Tensor] = None
    ) -> Tuple[Tensor, RS]:
        x_size0 = x.size(0)
        batch_size = hidden.size(0) // self.rate
        if x_size0 == batch_size:
            nsteps = 1
            outputs, hidden = self.forward_1step(x, hidden, masks)
            outputs = outputs.unsqueeze_(0)
        else:
            nsteps = x_size0 // batch_size
            inputs, masks = _reshape_batch(x, masks, nsteps)
            outputs, hidden = self.forward_nsteps(inputs, hidden, masks, nsteps)
        splitted_outputs = self._split_outputs(outputs)
        return splitted_outputs[:nsteps].view(x_size0, self.output_dim), hidden

    def _pad_inputs(self, inputs: Tensor, nsteps: int) -> Tuple[Tensor, int]:
        if nsteps % self.rate > 0:
            shape = inputs.shape
            dilated_nsteps = nsteps // self.rate + 1
            pad_dim = dilated_nsteps * self.rate - shape[0]
            zeros = torch.zeros(pad_dim, *shape[1:], device=inputs.device)
            return torch.cat((inputs, zeros))
        else:
            return inputs

    def _prepare_masks(self, masks: Tensor, nsteps: int) -> Tensor:
        reversed_ = (1.0 - masks).unsqueeze_(-1)
        dilated = self._prepare_inputs(reversed_, nsteps)
        return 1.0 - dilated.squeeze_()

    def _prepare_inputs(self, inputs: Tensor, nsteps: int) -> Tensor:
        rate = self.rate
        if nsteps % rate > 0:
            shape = inputs.shape
            dilated_nsteps = nsteps // rate + 1
            pad_dim = dilated_nsteps * rate - shape[0]
            zeros = torch.zeros(pad_dim, *shape[1:], device=inputs.device)
            padded_inputs = torch.cat((inputs, zeros))
        else:
            padded_inputs = inputs
        return torch.cat([padded_inputs[j::rate, :, :] for j in range(rate)], dim=1)

    def _split_outputs(self, dilated_outputs: Tensor) -> Tensor:
        batch_size = dilated_outputs.size(1) // self.rate
        blocks = [
            dilated_outputs[:, i * batch_size : (i + 1) * batch_size, :]
            for i in range(self.rate)
        ]
        interleaved = torch.stack(blocks).transpose(1, 0)
        time_steps = dilated_outputs.size(0) * self.rate
        return interleaved.reshape(time_steps, batch_size, -1)

    def initial_state(self, batch_size: int, device: Device) -> RS:
        return self.block.initial_state(batch_size * self.rate, device)
