# referenced https://github.com/chainer/chainerrl/blob/master/chainerrl/misc/random.py
import numpy as np
from typing import Set


def sample_indices(n: int, k: int) -> np.ndarray:
    if 3 * k >= n:
        return np.random.choice(n, k, replace=False)
    else:
        selected: Set = set()
        rands = np.random.randint(0, n, size=k * 2)
        j = k
        for i in range(k):
            x = rands[i]
            while x in selected:
                if j == 2 * k:
                    rands[k:] = np.random.randint(0, n, size=k)
                    j = k
                x = rands[i] = rands[j]
                j += 1
            selected.add(x)
        return rands[:k]
