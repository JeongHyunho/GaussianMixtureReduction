import pytest
import torch

from gmr.algo import fit_min_ise, ise_cost
from gmr.mixtures.utils import integral_prod_gauss_prob


def test_min_ise(plot, sampled_gm, helper):
    m_gm = fit_min_ise(sampled_gm, L=2)

    if plot:
        helper.sampled_gm_plot(m_gm)


@pytest.mark.parametrize('mode', ['cross', 'self'])
def test_batch_integral_prod_gauss_prob(mode, sampled_batch_gm):
    batch_p = integral_prod_gauss_prob(
        sampled_batch_gm.mu,
        sampled_batch_gm.var,
        sampled_batch_gm.mu,
        sampled_batch_gm.var,
        mode=mode,
    )

    for gm, bp_i in zip(sampled_batch_gm, batch_p):
        assert torch.all(torch.abs(bp_i - integral_prod_gauss_prob(gm.mu, gm.var, gm.mu, gm.var, mode=mode)) < 1e-6)


def test_batch_ise_cost(sampled_batch_gm):
    batch_c = ise_cost(sampled_batch_gm)

    for gm, bc_i in zip(sampled_batch_gm, batch_c):
        assert torch.all(bc_i == ise_cost(gm))


def test_batch_min_ise(sampled_batch_gm):
    m_batch_gm = fit_min_ise(sampled_batch_gm, L=2)

    for gm, m_gm in zip(sampled_batch_gm, m_batch_gm):
        assert m_gm == fit_min_ise(gm, L=2)
