import numpy as np
from .sample import RecurrentBatchSampler, sample_indices


def test_sample_large():
    arr = sample_indices(1000, 900)
    assert np.unique(arr).shape[0] == 900


def test_sample_small():
    arr = sample_indices(1000, 100)
    assert np.unique(arr).shape[0] == 100


def test_recurrent_batch_sampler():
    NSTEPS, NWORKERS = 12, 16
    r = RecurrentBatchSampler(NSTEPS, NWORKERS, 24)
    for batch in r:
        batch = batch.reshape(NSTEPS, -1)
        for i in range(NSTEPS - 1):
            before = batch[i]
            after = batch[i + 1]
            np.testing.assert_array_equal(before + NWORKERS, after)
