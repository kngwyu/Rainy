import numpy as np
import pytest
import torch
from .rms import RunningMeanStd, RunningMeanStdTorch
from ..utils import Device


@pytest.mark.parametrize('shape', [(10,), ()])
def test_rms(shape: tuple) -> None:
    dev = Device()
    dat = torch.randn(1000, dtype=torch.float64, device=dev.unwrapped)
    rms = RunningMeanStd(shape)
    rms_t = RunningMeanStdTorch(torch.Size(shape), device=dev)
    for i in range(10):
        d = dat[i * 100: i * 100 + 100]
        if len(shape) == 1:
            d = d.reshape(10, 10)
        rms.update(d.cpu().numpy())
        rms_t.update(d)
        np.testing.assert_almost_equal(rms.mean, rms_t.mean.detach().cpu().numpy())
        np.testing.assert_almost_equal(rms.var, rms_t.var.detach().cpu().numpy())
        np.testing.assert_almost_equal(rms.std(), rms_t.std().detach().cpu().numpy())
