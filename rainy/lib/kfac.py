from torch import nn
from torch.nn import functional as F
from torch.optim import Optimizer
import warnings


class KfacPreConditioner(Optimizer):
    def __init__(
            self,
            net: nn.Module,
            eps: float,
            avg_weight: float = 1.0,
            sua: bool = False,
            constraint_norm: bool = False,
    ) -> None:
        self.eps = eps
        self.avg_weight = avg_weight
        self.sua = sua
        self.constraint_norm = constraint_norm
        self.params = []
        for mod in net.modules():
            layer_type = type(mod).__name__
            if layer_type not in ['Linear', 'Conv2d']:
                warnings.warn('KfacPreConditioner doesn\'t support {}'.format(layer_type))
                continue
            mod.register_forward_pre_hookself(self._save_input)
            mod.register_backward_hook(self._save_grad_output)
            params = [mod.weight, mod.bias]
            self.params.append({'params': params, 'mod': mod, 'layer_type': layer_type})
        super().__init__(self.params, {})

    def _save_input(self, mod: nn.Module, ip: Tensor) -> None:
        """Save the ninputs of the layer
        """
        if mod.training:
            self.state[mod]['x'] = ip[0]

    def _save_grad(self, mod: nn.Module, _in: Tensor, grad_out: Tensor) -> None:
        """Save the output grads of the layer
        """
        if mod.training:
            self.state[mod]['gy'] = grad_out[0] * grad_out[0].size(0)

    def step(self) -> None:
        for group in self.param_groups:
            weight, bias = group['params']
            state = self.state[weight]

    def _calc_cov(self, group: dict, state: dict) -> Tensor:
        mod = group['mod']
        is_conv = group['layer_type'] == 'Conv2d'
        x, gy = self.state[mod]['x'], self.state[mod]['gy']
        if is_conv:
            x = x.view(x.shape[0], x.shape[1], -1)
        else:
            x = x.data.t()
