from enum import Enum
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from typing import Optional, Tuple, Union
import warnings


class Layer(Enum):
    LINEAR = 1
    CONV2D = 2

    def is_conv(self) -> bool:
        return self.value == self.LINEAR


def get_layer(mod: nn.Module) -> Union[Layer, str]:
    name = type(mod).__name__
    if name == 'Linear':
        return Layer.LINEAR
    elif name == 'Conv2d':
        return Layer.CONV2D
    else:
        return name


class KfacPreConditioner(Optimizer):
    def __init__(
            self,
            net: nn.Module,
            eps: float,
            alpha: float = 1.0,
            update_freq: int = 1,
            use_trace_norm_pi: bool = False,
            use_sua: bool = False,
            constraint_norm: bool = False,
    ) -> None:
        self.eps = eps
        self.alpha = alpha
        self.update_freq = update_freq
        self.use_trace_norm_pi = use_trace_norm_pi
        self.use_sua = use_sua
        self.constraint_norm = constraint_norm
        self.params = []
        self._counter = 0
        for mod in net.modules():
            layer_type = get_layer(mod)
            if not isinstance(layer_type, Layer):
                warnings.warn('KfacPreConditioner doesn\'t support {}. Skipped it.'
                              .format(layer_type))
                continue
            mod.register_forward_pre_hook(self._save_x)
            mod.register_backward_hook(self._save_g)
            params = [mod.weight, mod.bias]
            self.params.append({'params': params, 'mod': mod, 'layer_type': layer_type})
        super().__init__(self.params, {})

    def _save_x(self, mod: nn.Module, input_: Tuple[Tensor, ...]) -> None:
        """Save the inputs of the layer
        """
        if mod.training:
            self.state[mod]['x'], *_ = input_

    def _save_g(self, mod: nn.Module, _grad_in: tuple, grad_out: Tuple[Tensor, ...]) -> None:
        """Save the output gradients of the layer
        """
        if mod.training:
            grad_out, *_ = grad_out
            self.state[mod]['g'] = grad_out * grad_out.size(0)

    def step(self) -> None:
        fisher_norm = 0.0
        for group in self.param_groups:
            weight, bias = group['params']
            state = self.state[weight]
            self._update_stats(group, state)
        self._counter += 1

    def _update_stats(self, group: dict, state: dict) -> None:
        """Updates E[xxT], E[ggT], and their invs
        """
        xxt = self._compute_xxt(group, state)
        ggt = self._compute_ggt(group, state)
        if self.counter % self.update_freq != 0:
            return
        pi = self._compute_pi(xxt, ggt)
        eps = self.eps / state['num_locations']
        state['ixxt'] = self._compute_inv(xxt, (eps * pi) ** 0.5)
        state['iggt'] = self._compute_inv(ggt, (eps / pi) ** 0.5)

    def _update_params(
            self,
            weight: Tensor,
            bias: Optional[Tensor],
            layer: Layer,
            state: dict
    ) -> float:
        if layer.is_conv() and self.use_sua:
            pass

    def _computes_fisher(self, weight: Tensor, bias: Tensor, layer: Layer, state: dict):
        g = weight.grad.data
        if layer.is_conv():
            g = reshape(g.size(0), -1)

    def _compute_xxt(self, group: dict, state: dict) -> Tensor:
        """Computes E[x_i x_j^T] and memorize it
        """
        mod = group['mod']
        x = self.state[mod]['x']
        if group['layer_type'].is_conv():
            if self.use_sua:
                x = x.view(*x.shape[:2], -1)
            else:
                x = F.unfold(x, mod.kernel_size, padding=mod.padding, stride=mod.stride)
            x = x.data.transpose(1, 0).reshape(x.size(1), -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            x = torch.cat([x, torch.ones_like(x[:1])])
        return self._average(state, 'xxt', x)

    def _compute_ggt(self, group: dict, state: dict) -> Tensor:
        """Computes E[g_i g_j^T] and memorize it
        """
        g = self.state[group['mod']]['g']
        if group['layer_type'].is_conv():
            g = g.data.transpose(1, 0)
            state['num_locations'] = g.size(2) * g.size(3)
            g = g.reshape(g.size(0), -1)
        else:
            state['num_locations'] = 1
            g = g.data.t()
        return self._average(state, 'ggt', g)

    def _average(self, state: dict, param: str, mat: Tensor) -> Tensor:
        """Computes the moving average X <- (1 - α)X + αX' of E[xxT] or E[ggT]
        """
        if self._counter == 0:
            state[param] = torch.mm(mat, mat.t()) / float(mat.size(1))
        else:
            state[param].addmm_(
                mat1=mat,
                mat2=x.t(),
                beta=(1. - self.alpha),
                alpha=self.alpha / float(mat.size(1))
            )
        return state[param]

    def _compute_pi(self, xxt: Tensor, ggt: Tensor) -> float:
        """Computes π-correction for Tikhonov regularization
           TODO: Use different π for xxt & ggt
        """
        if self.use_trace_norm_pi:
            tx = torch.trace(xxt) * xxt.size(0)
            tg = torch.trace(ggt) * ggt.size(0)
            return tx.item() / tg.item()
        return 1.0

    def _compute_inv(self, mat: Tensor, eps: float) -> Tensor:
        """Computes (mat + πλ**0.5I) ^-1
        """
        diag = mat.new_full((mat.size(0),), eps)
        return (mat + torch.diag(diag)).inverse()
