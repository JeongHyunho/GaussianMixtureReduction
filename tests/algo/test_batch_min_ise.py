import numpy as np
import pytest

from algo.batch_min_ise import fit_batch_min_ise, batch_ise_cost
from algo.min_ise import fit_min_ise, ise_cost
from mixtures.batch_gm import BatchGM
from mixtures.utils import integral_prod_gauss_prob

batch_gm = BatchGM.sample_batch_gm(
    n=3,
    b=5,
    d=2,
    pi_alpha=5 * np.ones(3),
    mu_rng=[0., 3.],
    var_df=5,
    var_scale=1 / 5 * np.eye(2),
)


@pytest.mark.parametrize('mode', ['cross', 'self'])
def test_batch_integral_prod_gauss_prob(mode):
    batch_p = integral_prod_gauss_prob(batch_gm.mu, batch_gm.var, batch_gm.mu, batch_gm.var, mode=mode)

    for gm, bp_i in zip(batch_gm, batch_p):
        assert np.all(bp_i == integral_prod_gauss_prob(gm.mu, gm.var, gm.mu, gm.var, mode=mode))


def test_batch_ise_cost():
    batch_c = batch_ise_cost(batch_gm)

    for gm, bc_i in zip(batch_gm, batch_c):
        assert np.all(bc_i == ise_cost(gm))


def test_batch_min_ise():
    m_batch_gm = fit_batch_min_ise(batch_gm, L=2)

    for gm, m_gm in zip(batch_gm, m_batch_gm):
        assert m_gm == fit_min_ise(gm, L=2)
