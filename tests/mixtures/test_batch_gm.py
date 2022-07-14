from copy import deepcopy

import torch


def test_batch_gm_mul(sampled_batch_gm):
    gm_prod = sampled_batch_gm * sampled_batch_gm


def test_batch_gm_prob(sampled_batch_gm):
    t0, t1 = torch.meshgrid(torch.linspace(-1., 4., 100), torch.linspace(-1., 4., 100))
    sampled_batch_gm.prob(torch.stack([t0, t1], dim=-1).to(sampled_batch_gm.mu))


def test_batch_gm(sampled_batch_gm):
    bgm0 = sampled_batch_gm[0]
    bgm1 = sampled_batch_gm[:]
    bgm2 = sampled_batch_gm[1:2]


def test_batch_merge_gm(sampled_batch_gm):
    idx_list = [[0, 1], [1, 2], [0, 1], [1, 2], [0, 2]]
    cp_bgm = deepcopy(sampled_batch_gm)
    cp_bgm.merge(idx_list)

    for gm, m_gm, idx in zip(sampled_batch_gm, cp_bgm, idx_list):
        gm.merge([idx])
        assert m_gm == gm
