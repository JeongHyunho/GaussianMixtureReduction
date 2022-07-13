import numpy as np

from mixtures.batch_gm import batch_merge_gm, BatchGM
from mixtures.gm import merge_gm

batch_gm = BatchGM.sample_batch_gm(
    n=3,
    b=5,
    d=2,
    pi_alpha=5 * np.ones(3),
    mu_rng=[0., 3.],
    var_df=5,
    var_scale=1 / 5 * np.eye(2),
)


def test_batch_gm():
    bgm0 = batch_gm[0]
    bgm1 = batch_gm[:]
    bgm2 = batch_gm[1:2]


def test_merge_gm():
    idx_list = [[0, 1], [1, 2], [0, 1], [1, 2], [0, 2]]
    m_batch_gm = batch_merge_gm(batch_gm, idx_list)

    for gm, m_gm, idx in zip(batch_gm, m_batch_gm, idx_list):
        assert m_gm == merge_gm(gm, [idx])
