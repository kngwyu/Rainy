import torch
from torch import nn
from rainy.lib.kfac import KfacPreConditioner, Layer
import warnings


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


def test_kfac():
    in_shape = (10, 4, 8, 8)
    cnn = nn.Conv2d(4, 8, 4)
    fc = nn.Linear(8 * 5 * 5, 10)
    net = nn.Sequential(cnn, Flatten(), fc)
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        precond = KfacPreConditioner(net)
    out = net(torch.randn(*in_shape))
    loss = nn.MSELoss()(out, torch.randn(in_shape[0], 10))
    with precond.save_grad():
        loss.backward()
    precond.step()
    for group in precond.param_groups:
        state = precond.state[group['params'][0]]
        if group['layer_type'] is Layer.CONV2D:
            assert state['eg*ex'].shape == torch.Size((8, 65))
        else:
            assert state['eg*ex'].shape == torch.Size((10, 201))
