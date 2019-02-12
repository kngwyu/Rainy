from abc import ABC
from enum import Enum
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer
from typing import Optional, Tuple, Union
import warnings


class Layer(Enum):
    LINEAR = 1
    CONV2D = 2


def get_layer(mod: nn.Module) -> Union[Layer, str]:
    name = type(mod).__name__
    if name == 'Linear':
        return Layer.LINEAR
    elif name == 'Conv2d':
        return Layer.CONV2D
    else:
        return name


class PreConditioner(ABC, Optimizer):
    pass


class KfacPreConditioner(PreConditioner):
    def __init__(
            self,
            net: nn.Module,
            gamma: float = 1.0e-3,
            weight_decay: float = 0.0,
            tau: float = 100.0,
            update_freq: int = 10,
            use_trace_norm_pi: bool = True,
            use_sua: bool = False,
            constraint_norm: bool = False,
    ) -> None:
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.beta = math.exp(-1.0 / tau)
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
            fisher_norm += self._update_params(weight, bias, group['layer_type'], state)
            del self.state[group['mod']]['x']
            del self.state[group['mod']]['g']
        if self.constraint_norm:
            scale = (1. / fisher_norm) ** 0.5
            for group in self.param_groups:
                for param in filter(lambda p: p, group['params']):
                    param.grad.data *= scale
        self._counter += 1

    def _update_stats(self, group: dict, state: dict) -> None:
        """Updates E[xxT], E[ggT], and their invs
        """
        xxt = self.__xxt(group, state)
        ggt = self.__ggt(group, state)
        if self._counter % self.update_freq != 0:
            return
        pi = self.__pi(xxt, ggt)
        eps = (self.gamma + self.weight_decay) ** 0.5
        state['ixxt'] = self.__inv(xxt, eps * pi)
        state['iggt'] = self.__inv(ggt, eps / pi)

    def _update_params(
            self,
            weight: Tensor,
            bias: Optional[Tensor],
            layer: Layer,
            state: dict
    ) -> float:
        if layer is Layer.CONV2D and self.use_sua:
            raise NotImplementedError('SUA fisher is not yet implemented.')
        else:
            gw, gb = self.__fisher_grad(weight, bias, layer, state)
        fisher_norm = (weight.grad * gw).sum().item()
        weight.grad.data = gw
        if bias is not None:
            fisher_norm += (bias.grad * gb).sum().item()
            bias.grad.data = gb
        return fisher_norm

    def __fisher_grad(
            self,
            weight: Tensor,
            bias: Optional[Tensor],
            layer: Layer,
            state: dict
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Computes F^{-1}∇h
        """
        gw = weight.grad.data
        wshape = gw.shape
        if layer is Layer.CONV2D:
            gw = gw.reshape(wshape[0], -1)
        if bias is not None:
            gw = torch.cat([gw, bias.grad.data.view(-1, 1)], dim=1)
        gw = torch.mm(torch.mm(state['iggt'], gw), state['ixxt'])
        gb = None
        if bias is not None:
            gb = gw[:, -1].reshape(bias.shape)
            gw = gw[:, :-1]
        return gw.reshape(wshape), gb

    def __xxt(self, group: dict, state: dict) -> Tensor:
        """Computes E[x_i x_j^T] and memorize it
        """
        mod = group['mod']
        x = self.state[mod]['x']
        if group['layer_type'] is Layer.CONV2D:
            if self.use_sua:
                x = x.view(*x.shape[:2], -1)
            else:
                x = F.unfold(x, mod.kernel_size, padding=mod.padding, stride=mod.stride)
            x = x.data.transpose(1, 0).reshape(x.size(1), -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            x = torch.cat([x, torch.ones_like(x[:1])])
        return self.__average(state, 'xxt', x, float(x.size(1)))

    def __ggt(self, group: dict, state: dict) -> Tensor:
        """Computes E[g_i g_j^T] and memorize it
        """
        g = self.state[group['mod']]['g']
        scale = float(g.size(1))
        if group['layer_type'] is Layer.CONV2D:
            g = g.data.transpose(1, 0)
            scale *= g.size(2) * g.size(3)
            g = g.reshape(g.size(0), -1)
        else:
            g = g.data.t()
        return self.__average(state, 'ggt', g, float(scale))

    def __average(self, state: dict, param: str, mat: Tensor, scale: float) -> Tensor:
        """Computes the moving average X <- βX + (1-β)X'
        """
        if self._counter == 0:
            state[param] = torch.mm(mat, mat.t()) / scale
        else:
            state[param].addmm_(
                mat1=mat,
                mat2=x.t(),
                beta=self.beta,
                alpha=(1 - self.beta) / scale,
            )
        return state[param]

    def __pi(self, xxt: Tensor, ggt: Tensor) -> float:
        """Computes π-correction for Tikhonov regularization
           TODO: Use different π for xxt & ggt
        """
        if self.use_trace_norm_pi:
            tx = torch.trace(xxt) * ggt.size(0)
            tg = torch.trace(ggt) * xxt.size(0)
            return tx.item() / tg.item()
        return 1.0

    def __inv(self, mat: Tensor, eps: float) -> Tensor:
        """Computes (mat + εI) ^-1
        """
        diag = mat.new_full((mat.size(0),), eps)
        return (mat + torch.diag(diag)).inverse()
