from copy import deepcopy

import torch

from gmr.mixtures import cat_bgm


def test_batch_gm_mul(_batch_gm):
    gm_prod = _batch_gm * _batch_gm


def test_batch_gm_prob(_batch_gm):
    t0, t1 = torch.meshgrid(torch.linspace(-1., 4., 100), torch.linspace(-1., 4., 100))
    _batch_gm.prob(torch.stack([t0, t1], dim=-1).to(_batch_gm.mu))


def test_batch_gm_get_item(_batch_gm):
    bgm0 = _batch_gm[0]
    bgm1 = _batch_gm[:]
    bgm2 = _batch_gm[1:2]


def test_batch_merge_gm(_batch_gm):
    idx_list = [[0, 1], [1, 2], [0, 1], [1, 2], [0, 2]]
    cp_bgm = deepcopy(_batch_gm)
    cp_bgm.merge(idx_list)

    for gm, m_gm, idx in zip(_batch_gm, cp_bgm, idx_list):
        gm.merge([idx])
        assert m_gm == gm


def test_double_batch_gm_mul(_dbatch_gm):
    dbgm_prod = _dbatch_gm * _dbatch_gm


def test_double_batch_get_item(_dbatch_gm):
    bgm0 = _dbatch_gm[0]
    bgm1 = _dbatch_gm[:]
    bgm2 = _dbatch_gm[1, :]
    bgm3 = _dbatch_gm[1, 2]


def test_double_batch_merge_gm(_dbatch_gm):
    idx_list = [[[0, 1], [1, 2], [0, 1], [1, 2], [0, 2]],
                [[0, 1], [1, 2], [0, 1], [1, 2], [0, 2]]]
    cp_bgm = deepcopy(_dbatch_gm)
    cp_bgm.merge(idx_list)

    for b_gm, m_gm, idx in zip(_dbatch_gm, cp_bgm, idx_list):
        b_gm.merge(idx)
        assert m_gm == b_gm


def test_cat_bgm(_dbatch_gm):
    dbgm0 = deepcopy(_dbatch_gm)
    dbgm1 = deepcopy(_dbatch_gm)
    cat_dbgm = cat_bgm(dbgm0, dbgm1)
