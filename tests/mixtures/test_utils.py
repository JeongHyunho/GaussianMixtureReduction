import numpy as np

from mixtures.gm import GM
from mixtures.utils import integral_prod_gauss_prob, prod_gauss_dist

gm = GM.sample_gm(
        n=3,
        d=2,
        pi_alpha=5*np.ones(3),
        mu_rng=[0., 3.],
        var_df=5,
        var_scale=1/5 * np.eye(2),
    )


def test_integral_prod_gauss_prob():
    re_self = integral_prod_gauss_prob(gm.mu, gm.var, gm.mu, gm.var, mode='self')
    re_cross = integral_prod_gauss_prob(gm.mu, gm.var, gm.mu, gm.var, mode='cross')

    assert np.all(re_self == np.diag(re_cross))


def test_prod_gauss_dist():
    mu_self, var_self = prod_gauss_dist(gm.mu, gm.var, gm.mu, gm.var, mode='self')
    mu_cross, var_cross = prod_gauss_dist(gm.mu, gm.var, gm.mu, gm.var, mode='cross')

    assert np.all(mu_self == mu_cross[np.arange(gm.n), np.arange(gm.n), :])
    assert np.all(var_self == var_cross[np.arange(gm.n), np.arange(gm.n), :, :])
