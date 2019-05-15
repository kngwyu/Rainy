import numpy as np
import torch
from .rms import RunningMeanStd, RunningMeanStdTorch
from ..utils import Device


def test_rms() -> None:
    dev = Device()
    dat = torch.randn(1000, dtype=torch.float64, device=dev.unwrapped)
    rms = RunningMeanStd((10,))
    rms_t = RunningMeanStdTorch(torch.Size((10,)), device=dev)
    for i in range(10):
        d = dat[i * 100: i * 100 + 100].reshape(10, 10)
        rms.update(d.cpu().numpy())
        rms_t.update(d)
        np.testing.assert_almost_equal(rms.mean, rms_t.mean.cpu().numpy())
        np.testing.assert_almost_equal(rms.var, rms_t.var.cpu().numpy())
        np.testing.assert_almost_equal(rms.std(), rms_t.std().cpu().numpy())
