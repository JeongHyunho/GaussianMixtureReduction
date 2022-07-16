import torch

from gmr.mixtures.utils import integral_prod_gauss_prob, prod_gauss_dist


def test_integral_prod_gauss_prob(_gm):
    re_self = integral_prod_gauss_prob(
        _gm.mu, _gm.var, _gm.mu, _gm.var, mode='self')
    re_cross = integral_prod_gauss_prob(
        _gm.mu, _gm.var, _gm.mu, _gm.var, mode='cross')

    assert torch.all(torch.abs(re_self - torch.diag(re_cross)) < 1e-6)


def test_prod_gauss_dist(_gm):
    mu_self, var_self = prod_gauss_dist(
        _gm.mu, _gm.var, _gm.mu, _gm.var, mode='self')
    mu_cross, var_cross = prod_gauss_dist(
        _gm.mu, _gm.var, _gm.mu, _gm.var, mode='cross')

    idx = torch.arange(_gm.n)
    assert torch.all(torch.abs(mu_self - mu_cross[idx, idx, :]) < 1e-6)
    assert torch.all(torch.abs(var_self - var_cross[idx, idx, :, :]) < 1e-6)
