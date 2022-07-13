from copy import deepcopy

import numpy as np

from mixtures.batch_gm import BatchGM

batch_gm = BatchGM.sample_batch_gm(
    n=3,
    b=5,
    d=2,
    pi_alpha=5 * np.ones(3),
    mu_rng=[0., 3.],
    var_df=5,
    var_scale=1 / 5 * np.eye(2),
)


def test_batch_gm_mul():
    gm_prod = batch_gm * batch_gm


def test_batch_gm_prob():
    t0, t1 = np.meshgrid(np.linspace(-1., 4., 100), np.linspace(-1., 4., 100))
    batch_gm.prob(np.stack([t0, t1], axis=-1))


def test_batch_gm():
    bgm0 = batch_gm[0]
    bgm1 = batch_gm[:]
    bgm2 = batch_gm[1:2]


def test_merge_gm():
    idx_list = [[0, 1], [1, 2], [0, 1], [1, 2], [0, 2]]
    cp_bgm = deepcopy(batch_gm)
    cp_bgm.merge(idx_list)

    for gm, m_gm, idx in zip(batch_gm, cp_bgm, idx_list):
        gm.merge([idx])
        assert m_gm == gm
