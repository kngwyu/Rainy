import torch
from torch import Tensor
from .dqn import DqnAgent
from ..prelude import Array


class DoubleDqnAgent(DqnAgent):
    @torch.no_grad()
    def _q_next(self, next_states: Array) -> Tensor:
        """Returns Q values of next_states, supposing torch.no_grad() is called
        """
        q_next = self.target_net(next_states)
        q_values = self.net.q_values(next_states, nostack=True)
        return q_next[self.batch_indices, q_values.argmax(dim=-1)]
