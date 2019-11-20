from abc import ABC, abstractmethod
from contextlib import contextmanager
from enum import Enum
import math
import torch
from torch import nn, Tensor
from torch.nn import functional as F
from torch.optim import Optimizer, SGD
from typing import Any, Callable, Generator, List, Optional, Tuple, Union
import warnings
from ..prelude import Params


class Layer(Enum):
    LINEAR = 1
    CONV2D = 2


def get_layer(mod: nn.Module) -> Union[Layer, str]:
    name = type(mod).__name__
    if name == "Linear":
        return Layer.LINEAR
    elif name == "Conv2d":
        return Layer.CONV2D
    else:
        return name


def default_sgd(
    eta_max: float = 0.25, momentum: float = 0.9
) -> Callable[[Params], Optimizer]:
    """Returns the SGD optimizer which has the default setting for used with K-FAC.
    """

    def _sgd(params: Params) -> Optimizer:
        return SGD(params, lr=eta_max * (1.0 - momentum), momentum=momentum)

    return _sgd


class NormScaler(ABC):
    @abstractmethod
    def scale(self, fisher_norm: float) -> float:
        pass

    def __call__(self, param_groups: List[dict], fisher_norm: float) -> None:
        scale = self.scale(fisher_norm)
        for group in param_groups:
            for param in group["params"]:
                if param is not None:
                    param.grad.data.mul_(scale)


class SquaredFisherScaler(NormScaler):
    """Section 5 in https://jimmylba.github.io/papers/nsync.pdf
    """

    def __init__(self, eta_max: float = 0.25, delta: float = 0.001) -> None:
        self.eta_max2 = eta_max ** 2
        self.delta = delta

    def scale(self, fisher_norm: float) -> float:
        return min(1.0, math.sqrt(self.delta / (fisher_norm * self.eta_max2)))


class DiagonalScaler(NormScaler):
    """https://arxiv.org/abs/1705.09319
    """

    def __init__(self, mu: float = 0.001) -> None:
        self.mu = mu

    def scale(self, fisher_norm: float) -> float:
        return math.sqrt(1.0 / (fisher_norm + self.mu))


class DummyScaler(NormScaler):
    def scale(self, fisher_norm: float) -> float:
        return 1.0


class PreConditioner(ABC, Optimizer):
    pass


class KfacPreConditioner(PreConditioner):
    def __init__(
        self,
        net: nn.Module,
        damping: float = 1.0e-4,
        weight_decay: float = 0.0,
        tau: float = 100.0,
        eps: float = 1.0e-6,
        update_freq: int = 2,
        norm_scaler: NormScaler = SquaredFisherScaler(),
    ) -> None:
        self.gamma = (damping + weight_decay) ** 0.5
        self.beta = math.exp(-1.0 / tau)
        self.eps = eps
        self.update_freq = update_freq
        self.norm_scaler = norm_scaler
        self.params: List[dict] = []
        self._counter = 0
        self._save_grad = False
        for mod in net.modules():
            layer_type = get_layer(mod)
            if not isinstance(layer_type, Layer):
                warnings.warn(
                    "KfacPreConditioner doesn't support {}. Skipped it.".format(
                        layer_type
                    )
                )
                continue
            mod.register_forward_pre_hook(self._save_x)
            mod.register_backward_hook(self._save_g)
            params = [mod.weight]
            if mod.bias is not None:
                params.append(mod.bias)
            self.params.append({"params": params, "mod": mod, "layer_type": layer_type})
        super().__init__(self.params, {})

    def with_saving_grad(self, f: Callable[[], Any]) -> Any:
        self._save_grad = True
        res = f()
        self._save_grad = False
        return res

    def _save_x(self, mod: nn.Module, input_: Tuple[Tensor, ...]) -> None:
        """Save the inputs of the layer
        """
        if mod.training:
            self.state[mod]["x"], *_ = input_

    def _save_g(
        self, mod: nn.Module, _grad_in: tuple, grad_out: Tuple[Tensor, ...]
    ) -> None:
        """Save the output gradients of the layer
        """
        if mod.training and self._save_grad:
            grad, *_ = grad_out
            self.state[mod]["g"] = grad * grad.size(0)

    def step(self) -> None:
        fisher_norm = 0.0
        for group in self.param_groups:
            weight, bias = group["params"]
            state = self.state[weight]
            self.__xxt(group, state)
            self.__ggt(group, state)
            if self._counter % self.update_freq == 0:
                self.__eigend(state)
            fisher_norm += self._update_params(weight, bias, group["layer_type"], state)
            del self.state[group["mod"]]["x"], self.state[group["mod"]]["g"]
        self.norm_scaler(self.param_groups, fisher_norm)
        self._counter += 1

    def _update_params(
        self, weight: Tensor, bias: Optional[Tensor], layer: Layer, state: dict
    ) -> float:
        """Updates gradients
        """
        gw, gb = self.__fisher_grad(weight, bias, layer, state)
        fisher_norm = weight.grad.mul(gw).sum().item()
        weight.grad.data.copy_(gw)
        if bias is not None:
            fisher_norm += bias.grad.mul(gb).sum().item()
            bias.grad.data.copy_(gb)
        return fisher_norm

    def __fisher_grad(
        self, weight: Tensor, bias: Optional[Tensor], layer: Layer, state: dict
    ) -> Tuple[Tensor, Optional[Tensor]]:
        """Computes F^{-1}∇h
        """
        grad = weight.grad.data
        if layer is Layer.CONV2D:
            grad = grad.view(grad.size(0), -1)
        if bias is not None:
            grad = torch.cat([grad, bias.grad.data.view(-1, 1)], dim=1)
        v1 = torch.chain_matmul(state["vg"].t(), grad, state["vx"])
        v2 = v1.div_(state["eg*ex"].add(self.gamma))
        grad = torch.chain_matmul(state["vg"], v2, state["vx"].t())
        gb = None
        if bias is not None:
            gb = grad[:, -1].reshape(bias.shape)
            gw = grad[:, :-1]
        return gw.reshape(weight.shape), gb

    def __eigend(self, state: dict) -> None:
        """Computes eigen decomposition of E[xx] and E[gg]
        """
        ex, state["vx"] = torch.symeig(state["xxt"], eigenvectors=True)
        eg, state["vg"] = torch.symeig(state["ggt"], eigenvectors=True)
        state["eg*ex"] = torch.ger(eg.clamp_(min=self.eps), ex.clamp_(min=self.eps))

    def __xxt(self, group: dict, state: dict) -> None:
        """Computes E[xi xj]^T and memorize it
        """
        mod = group["mod"]
        x = self.state[mod]["x"]
        if group["layer_type"] is Layer.CONV2D:
            x = F.unfold(x, mod.kernel_size, padding=mod.padding, stride=mod.stride)
            x = x.data.transpose(1, 0).reshape(x.size(1), -1)
        else:
            x = x.data.t()
        if mod.bias is not None:
            x = torch.cat([x, torch.ones_like(x[:1])])
        self.__average(state, "xxt", x, float(x.size(1)))

    def __ggt(self, group: dict, state: dict) -> None:
        """Computes E[gi gj]^T and memorize it
        """
        g = self.state[group["mod"]]["g"]
        if group["layer_type"] is Layer.CONV2D:
            g = g.data.transpose(1, 0).reshape(g.size(1), -1)
        else:
            g = g.data.t()
        self.__average(state, "ggt", g, float(g.size(1)))

    def __average(self, state: dict, param: str, mat: Tensor, scale: float) -> None:
        """Computes the moving average X <- βX + (1-β)X'
        """
        if self._counter == 0:
            state[param] = torch.mm(mat, mat.t()).div_(scale)
        else:
            state[param].addmm_(
                mat1=mat, mat2=mat.t(), beta=self.beta, alpha=(1 - self.beta) / scale,
            )

    @contextmanager
    def save_grad(self) -> Generator[None, None, None]:
        self._save_grad = True
        yield
        self._save_grad = False
