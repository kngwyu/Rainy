"""From OpenAI baselines:
https://github.com/openai/baselines/blob/master/baselines/common/running_mean_std.py
"""
from typing import Sequence, Tuple

import numpy as np
import torch

from ..lib import mpi
from ..prelude import Array, Self
from ..utils import Device
from ..utils.state_dict import TensorStateDict


class RunningMeanStd:
    """Calcurate running mean and variance"""

    def __init__(self, shape: Sequence[int], epsilon: float = 1.0e-4) -> None:
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = epsilon

    def update(self, x: Array[float]) -> None:
        x_mean, x_var = mpi.array_mean_and_var(x)
        self.mean, self.var, self.count = _update_rms(
            self.mean,
            self.var,
            self.count,
            x_mean,
            x_var,
            x.shape[0] * mpi.global_size(),
        )

    def std(self, eps: float = 1.0e-8) -> Array[float]:
        return np.sqrt(self.var + eps)

    def copyto(self, other: Self) -> None:
        np.copyto(other.mean, self.mean)
        np.copyto(other.var, self.var)

    def __repr__(self) -> str:
        return f"RMS(mean: {self.mean}, var: {self.var})"


def _update_rms(
    mean: Array[float],
    var: Array[float],
    count: float,
    batch_mean: float,
    batch_var: float,
    batch_count: int,
) -> Tuple[Array[float], Array[float], float]:
    """https://en.wikipedia.org/wiki/Algorithms_for_calculating_variance#Parallel_algorithm"""
    delta = batch_mean - mean
    tot_count = count + batch_count
    new_mean = np.add(mean, delta * batch_count / tot_count)
    m_a = var * count
    m_b = batch_var * batch_count
    M2 = m_a + m_b + np.square(delta) * count * batch_count / tot_count
    new_var = M2 / tot_count
    new_count = tot_count
    return np.asarray(new_mean), np.asarray(new_var), new_count


class RunningMeanStdTorch(TensorStateDict):
    """Same as RunningMeanStd, but uses PyTorch Tensor"""

    def __init__(
        self, shape: torch.Size, device: Device, epsilon: float = 1.0e-4
    ) -> None:
        self.mean = device.zeros(shape, dtype=torch.float64)
        self.var = device.ones(shape, dtype=torch.float64)
        self.count = torch.tensor(epsilon, dtype=torch.float64, device=device.unwrapped)
        self.device = device

    @torch.no_grad()
    def update(self, x: torch.Tensor) -> None:
        x_mean, x_var = mpi.tensor_mean_and_var(x)
        _update_rms_torch(
            self.mean,
            self.var,
            self.count,
            x_mean,
            x_var,
            torch.tensor(x.size(0) * mpi.global_size(), device=self.device.unwrapped),
        )

    def std(self, eps: float = 1.0e-8) -> torch.Tensor:
        return torch.sqrt(self.var + eps)


def _update_rms_torch(mean, var, count, batch_mean, batch_var, batch_count):
    """Inplace, PyTorch implementation of _update_rms"""
    delta = batch_mean - mean
    tot_count = count + batch_count
    mean.add_(delta * batch_count / tot_count)
    m_b = batch_var * batch_count
    delta.pow_(2).mul_(count).mul_(batch_count).div_(tot_count)
    var.mul_(count).add_(m_b).add_(delta).div_(tot_count)
    count.add_(batch_count)
