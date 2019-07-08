import numpy as np
from .sample import RecurrentBatchSampler, sample_indices


def test_sample_large():
    arr = sample_indices(100, 90)
    assert np.unique(arr).__len__() == 90


def test_sample_small():
    arr = sample_indices(100, 30)
    assert np.unique(arr).__len__() == 30


def test_recurrent_batch_sampler():
    NSTEPS, NWORKERS = 12, 16
    r = RecurrentBatchSampler(NSTEPS, NWORKERS, 24)
    for batch in r:
        batch = batch.reshape(NSTEPS, -1)
        for i in range(NSTEPS - 1):
            before = batch[i]
            after = batch[i + 1]
            np.testing.assert_array_equal(before + NWORKERS, after)
