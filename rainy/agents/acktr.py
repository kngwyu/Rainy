"""
This module has an implementation of ACKTR, which is described in
- Scalable trust-region method for deep reinforcement learning using
 Kronecker-factored approximation
  - URL: https://arxiv.org/abs/1708.05144
"""
import torch

from ..config import Config
from ..net import Policy
from .a2c import A2CAgent


class ACKTRAgent(A2CAgent):
    def __init__(self, config: Config) -> None:
        super().__init__(config)
        if self.net.is_recurrent:
            raise NotImplementedError("K-FAC for RNN is not implemented!")
        self.precond = config.preconditioner(self.net)

    def _pre_backward(self, policy: Policy, value: torch.Tensor) -> None:
        """Calculate emprical fisher loss
        """
        self.net.zero_grad()
        policy_fisher_loss = -policy.log_prob().mean()
        sample_value = torch.randn_like(value) + value.detach()
        value_fisher_loss = -(value - sample_value).pow(2).mean()
        fisher_loss = policy_fisher_loss + value_fisher_loss
        with self.precond.save_grad():
            fisher_loss.backward(retain_graph=True)

    def _step_optimizer(self) -> None:
        """Approximates F^-1∇h and apply it.
        """
        self.precond.step()
        self.optimizer.step()
        self.lr_cooler.lr_decay(self.optimizer)
