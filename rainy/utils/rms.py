"""From https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
"""
import numpy as np
import torch
from typing import Tuple
from ..prelude import Array
from ..utils import Device
from ..utils.state_dict import TensorStateDict


class RunningMeanStd:
    """Calcurate running mean and variance
    """
    def __init__(self, shape: Tuple[int, ...], epsilon: float = 1.0e-4) -> None:
        self.mean = np.zeros(shape, 'float64')
        self.var = np.ones(shape, 'float64')
        self.count = epsilon

    def update(self, x: Array[float]) -> None:
        self.mean, self.var, self.count = _update_rms(
            self.mean,
            self.var,
            self.count,
            np.mean(x, axis=0),
            np.var(x, axis=0),
            x.shape[0]
        )

    def std(self, eps: float = 1.0e-8) -> Array[float]:
        return np.sqrt(self.var + eps)


def _update_rms(mean, var, count, batch_mean, batch_var, batch_count):
    """https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm
    """
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = mean + delta * batch_count / tot_count
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return new_mean, new_var, new_count


class RunningMeanStdTorch(TensorStateDict):
    """Same as RunningMeanStd, but uses PyTorch Tensor
    """
    def __init__(self, shape: torch.Size, device: Device, epsilon: float = 1.0e-4) -> None:
        self.mean = device.zeros(shape, dtype=torch.float64)
        self.var = device.ones(shape, dtype=torch.float64)
        self.count = torch.tensor(epsilon, dtype=torch.float64, device=device.unwrapped)
        self.device = device

    def update(self, x: torch.Tensor) -> None:
        with torch.no_grad():
            _update_rms_torch(
                self.mean,
                self.var,
                self.count,
                x.mean(dim=0),
                x.var(dim=0, unbiased=False),
                torch.tensor(x.size(0), device=self.device.unwrapped)
            )

    def std(self, eps: float = 1.0e-8) -> torch.Tensor:
        return torch.sqrt(self.var + eps)


@torch.jit.script
def _update_rms_torch(mean, var, count, batch_mean, batch_var, batch_count):
    """PyTorch version of _update_rms
    """
    delta = batch_mean - mean
    tot_count = count + batch_count
    mean.add_(delta * batch_count / tot_count)
    m_b = batch_var * batch_count
    delta.pow_(2).mul_(count).mul_(batch_count).div_(tot_count)
    var.mul_(count).add_(m_b).add_(delta).div_(tot_count)
    count.add_(batch_count)
