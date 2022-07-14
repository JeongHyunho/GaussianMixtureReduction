import torch

from mixtures.utils import integral_prod_gauss_prob, prod_gauss_dist


def test_integral_prod_gauss_prob(sampled_gm):
    re_self = integral_prod_gauss_prob(
        sampled_gm.mu, sampled_gm.var, sampled_gm.mu, sampled_gm.var, mode='self')
    re_cross = integral_prod_gauss_prob(
        sampled_gm.mu, sampled_gm.var, sampled_gm.mu, sampled_gm.var, mode='cross')

    assert torch.all(re_self == torch.diag(re_cross))


def test_prod_gauss_dist(sampled_gm):
    mu_self, var_self = prod_gauss_dist(
        sampled_gm.mu, sampled_gm.var, sampled_gm.mu, sampled_gm.var, mode='self')
    mu_cross, var_cross = prod_gauss_dist(
        sampled_gm.mu, sampled_gm.var, sampled_gm.mu, sampled_gm.var, mode='cross')

    idx = torch.arange(sampled_gm.n)
    assert torch.all(mu_self == mu_cross[idx, idx, :])
    assert torch.all(var_self == var_cross[idx, idx, :, :])
