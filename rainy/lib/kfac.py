from abc import ABC
from enum import Enum
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer, SGD
from typing import Callable, Optional, Tuple, Union
import warnings
from ..prelude import Params


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


def default_sgd(eta_max: float = 0.25, momentum: float = 0.9) -> Callable[[Params], Optimizer]:
    """Returns the SGD optimizer which has the default setting for used with K-FAC.
    """
    def _sgd(params: Params) -> Optimizer:
        return SGD(params, lr=eta_max * (1.0 - momentum), momentum=momentum)
    return _sgd


class PreConditioner(ABC, Optimizer):
    pass


class KfacPreConditioner(PreConditioner):
    def __init__(
            self,
            net: nn.Module,
            gamma: float = 1.0e-3,
            weight_decay: float = 0.0,
            tau: float = 100.0,
            eta_max: float = 0.25,
            delta: float = 0.001,
            update_freq: int = 2,
            use_trace_norm_pi: bool = True,
            constraint_norm: bool = True,
            use_sua: bool = False,
    ) -> None:
        self.gamma = gamma
        self.weight_decay = weight_decay
        self.beta = math.exp(-1.0 / tau)
        self.eta_max = eta_max
        self.delta = delta
        self.update_freq = update_freq
        self.use_trace_norm_pi = use_trace_norm_pi
        self.constraint_norm = constraint_norm
        self.use_sua = use_sua
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
            del self.state[group['mod']]['x'], self.state[group['mod']]['g']
        if self.constraint_norm:
            self._scale_norm(fisher_norm)
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
        """Updates gradients
        """
        if layer is Layer.CONV2D and self.use_sua:
            raise NotImplementedError('SUA fisher is not yet implemented.')
        else:
            gw, gb = self.__fisher_grad(weight, bias, layer, state)
        fisher_norm = (weight.grad * gw).sum().item()
        weight.grad.data.copy_(gw)
        if bias is not None:
            fisher_norm += (bias.grad * gb).sum().item()
            bias.grad.data.copy_(gb)
        return fisher_norm

    def _scale_norm(self, fisher_norm: float) -> None:
        scale = (1.0 / fisher_norm) ** 0.5
        for group in self.param_groups:
            for param in filter(lambda p: p is not None, group['params']):
                param.grad.data.mul_(scale)

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
        scale = float(g.size(0))
        if group['layer_type'] is Layer.CONV2D:
            scale *= g.size(2) * g.size(3)
            g = g.data.transpose(1, 0).reshape(g.size(1), -1)
        else:
            g = g.data.t()
        return self.__average(state, 'ggt', g, float(scale))

    def __average(self, state: dict, param: str, mat: Tensor, scale: float) -> Tensor:
        """Computes the moving average X <- βX + (1-β)X'
        """
        if self._counter == 0:
            state[param] = torch.mm(mat, mat.t()).div_(scale)
        else:
            state[param].addmm_(
                mat1=mat,
                mat2=mat.t(),
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


class KFAC(Optimizer):
    def __init__(self, net, eps, sua=False, pi=False, update_freq=1,
                 alpha=0.01, constraint_norm=False):
        """ K-FAC Preconditionner for Linear and Conv2d layers.

        Computes the K-FAC of the second moment of the gradients.
        It works for Linear and Conv2d layers and silently skip other layers.

        Args:
            net (torch.nn.Module): Network to precondition.
            eps (float): Tikhonov regularization parameter for the inverses.
            sua (bool): Applies SUA approximation.
            pi (bool): Computes pi correction for Tikhonov regularization.
            update_freq (int): Perform inverses every update_freq updates.
            alpha (float): Running average parameter (if == 1, no r. ave.).
            constraint_norm (bool): Scale the gradients by the squared
                fisher norm.
        """
        self.eps = eps
        self.sua = sua
        self.pi = pi
        self.update_freq = update_freq
        self.alpha = alpha
        self.constraint_norm = constraint_norm
        self.params = []
        self._iteration_counter = 0
        for mod in net.modules():
            mod_class = mod.__class__.__name__
            if mod_class in ['Linear', 'Conv2d']:
                mod.register_forward_pre_hook(self._save_input)
                mod.register_backward_hook(self._save_grad_output)
                params = [mod.weight]
                if mod.bias is not None:
                    params.append(mod.bias)
                d = {'params': params, 'mod': mod, 'layer_type': mod_class}
                self.params.append(d)
        super(KFAC, self).__init__(self.params, {})

    def step(self, update_stats=True, update_params=True):
        """Performs one step of preconditioning."""
        fisher_norm = 0.
        for group in self.param_groups:
            # Getting parameters
            if len(group['params']) == 2:
                weight, bias = group['params']
            else:
                weight = group['params'][0]
                bias = None
            state = self.state[weight]
            # Update convariances and inverses
            if update_stats:
                if self._iteration_counter % self.update_freq == 0:
                    self._compute_covs(group, state)
                    ixxt, iggt = self._inv_covs(state['xxt'], state['ggt'],
                                                state['num_locations'])
                    state['ixxt'] = ixxt
                    state['iggt'] = iggt
                else:
                    if self.alpha != 1:
                        self._compute_covs(group, state)
            if update_params:
                # Preconditionning
                gw, gb = self._precond(weight, bias, group, state)
                # Updating gradients
                if self.constraint_norm:
                    fisher_norm += (weight.grad * gw).sum()
                weight.grad.data = gw
                if bias is not None:
                    if self.constraint_norm:
                        fisher_norm += (bias.grad * gb).sum()
                    bias.grad.data = gb
            # Cleaning
            if 'x' in self.state[group['mod']]:
                del self.state[group['mod']]['x']
            if 'gy' in self.state[group['mod']]:
                del self.state[group['mod']]['gy']
        # Eventually scale the norm of the gradients
        if update_params and self.constraint_norm:
            scale = (1. / fisher_norm) ** 0.5
            for group in self.param_groups:
                for param in group['params']:
                    param.grad.data *= scale
        if update_stats:
            self._iteration_counter += 1

    def _save_input(self, mod, i):
        """Saves input of layer to compute covariance."""
        if mod.training:
            self.state[mod]['x'] = i[0]

    def _save_grad_output(self, mod, grad_input, grad_output):
        """Saves grad on output of layer to compute covariance."""
        if mod.training:
            self.state[mod]['gy'] = grad_output[0] * grad_output[0].size(0)

    def _precond(self, weight, bias, group, state):
        """Applies preconditioning."""
        if group['layer_type'] == 'Conv2d' and self.sua:
            return self._precond_sua(weight, bias, group, state)
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad.data
        s = g.shape
        if group['layer_type'] == 'Conv2d':
            g = g.contiguous().view(s[0], s[1]*s[2]*s[3])
        if bias is not None:
            gb = bias.grad.data
            g = torch.cat([g, gb.view(gb.shape[0], 1)], dim=1)
        g = torch.mm(torch.mm(iggt, g), ixxt)
        if group['layer_type'] == 'Conv2d':
            g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1].contiguous().view(*bias.shape)
            g = g[:, :-1]
        else:
            gb = None
        g = g.contiguous().view(*s)
        return g, gb

    def _precond_sua(self, weight, bias, group, state):
        """Preconditioning for KFAC SUA."""
        ixxt = state['ixxt']
        iggt = state['iggt']
        g = weight.grad.data
        s = g.shape
        mod = group['mod']
        g = g.permute(1, 0, 2, 3).contiguous()
        if bias is not None:
            gb = bias.grad.view(1, -1, 1, 1).expand(1, -1, s[2], s[3])
            g = torch.cat([g, gb], dim=0)
        g = torch.mm(ixxt, g.contiguous().view(-1, s[0]*s[2]*s[3]))
        g = g.view(-1, s[0], s[2], s[3]).permute(1, 0, 2, 3).contiguous()
        g = torch.mm(iggt, g.view(s[0], -1)).view(s[0], -1, s[2], s[3])
        g /= state['num_locations']
        if bias is not None:
            gb = g[:, -1, s[2]//2, s[3]//2]
            g = g[:, :-1]
        else:
            gb = None
        return g, gb

    def _compute_covs(self, group, state):
        """Computes the covariances."""
        mod = group['mod']
        x = self.state[group['mod']]['x']
        gy = self.state[group['mod']]['gy']
        # Computation of xxt
        if group['layer_type'] == 'Conv2d':
            if not self.sua:
                x = F.unfold(x, mod.kernel_size, padding=mod.padding,
                             stride=mod.stride)
            else:
                x = x.view(x.shape[0], x.shape[1], -1)
            x = x.data.permute(1, 0, 2).contiguous().view(x.shape[1], -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            ones = torch.ones_like(x[:1])
            x = torch.cat([x, ones], dim=0)
        if self._iteration_counter == 0:
            state['xxt'] = torch.mm(x, x.t()) / float(x.shape[1])
        else:
            state['xxt'].addmm_(mat1=x, mat2=x.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(x.shape[1]))
        # Computation of ggt
        if group['layer_type'] == 'Conv2d':
            gy = gy.data.permute(1, 0, 2, 3)
            state['num_locations'] = gy.shape[2] * gy.shape[3]
            gy = gy.contiguous().view(gy.shape[0], -1)
        else:
            gy = gy.data.t()
            state['num_locations'] = 1
        if self._iteration_counter == 0:
            state['ggt'] = torch.mm(gy, gy.t()) / float(gy.shape[1])
        else:
            state['ggt'].addmm_(mat1=gy, mat2=gy.t(),
                                beta=(1. - self.alpha),
                                alpha=self.alpha / float(gy.shape[1]))

    def _inv_covs(self, xxt, ggt, num_locations):
        """Inverses the covariances."""
        # Computes pi
        pi = 1.0
        if self.pi:
            tx = torch.trace(xxt) * ggt.shape[0]
            tg = torch.trace(ggt) * xxt.shape[0]
            pi = (tx / tg)
        # Regularizes and inverse
        eps = self.eps / num_locations
        diag_xxt = xxt.new(xxt.shape[0]).fill_((eps * pi) ** 0.5)
        diag_ggt = ggt.new(ggt.shape[0]).fill_((eps / pi) ** 0.5)
        ixxt = (xxt + torch.diag(diag_xxt)).inverse()
        iggt = (ggt + torch.diag(diag_ggt)).inverse()
        return ixxt, iggt
