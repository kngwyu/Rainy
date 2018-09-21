from .strategies import EpsGreedy
from .cooler import LinearCooler
import numpy as np
from ..net import value_net
from ..util import Device


def test_eps_greedy():
    eg = EpsGreedy(1.0, LinearCooler(1.0, 0.1, int(100)), value_net.fc((100,), 10, Device()))
    for i in range(0, 100):
        act = eg.select_action(np.arange(100))
    assert(eg.epsilon == 0.1)
