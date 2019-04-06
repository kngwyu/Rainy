from itertools import chain
import numpy as np
from torch.utils.data.sampler import Sampler
from typing import Iterable, List
from ..prelude import Array


def sample_indices(n: int, k: int) -> np.ndarray:
    """Sample k numbers from [0, n)
       Based on https://github.com/chainer/chainerrl/blob/master/chainerrl/misc/random.py
    """
    if 3 * k >= n:
        return np.random.choice(n, k, replace=False)
    else:
        selected = np.repeat(False, n)
        rands = np.random.randint(0, n, size=k * 2)
        j = k
        for i in range(k):
            x = rands[i]
            while selected[x]:
                if j == 2 * k:
                    rands[k:] = np.random.randint(0, n, size=k)
                    j = k
                x = rands[i] = rands[j]
                j += 1
            selected[x] = True
        return rands[:k]


class OrderedBatchSampler(Sampler):
    def __init__(self, nsteps: int, nworkers: int, batch_size: int) -> None:
        if batch_size % nsteps > 0:
            raise ValueError('batch_size must be a multiple of nsteps')
        self.nsteps = nsteps
        self.nworkers = nworkers
        self.batch_size = batch_size

    def __iter__(self) -> Iterable[Array[int]]:
        env_num = self.batch_size // self.nsteps
        perm = np.random.permutation(self.nworkers)
        for i in range(self.nworkers // env_num):
            stop, step = self.nsteps * self.nworkers, self.nsteps
            yield np.concatenate([np.arange(w, stop, step) for w in perm[i:i + env_num]])

    def __len__(self) -> int:
        return (self.nsteps * self.nworkers) // self.batch_size

