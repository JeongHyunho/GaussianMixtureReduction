import numpy as np
from copy import deepcopy

from mixtures.gm import GM, merge_gm
from mixtures.utils import integral_prod_gauss_prob


def fit_min_ise(gm_ori: GM, L: int):
    """Iteratively merge two components of mixture which incurs minimum ise increase

    Args:
        gm_ori: the gaussian mixture to be reduced
        L: the number of components of reduced mixture

    Returns:
        GM: reduced gaussian mixture

    """

    out_gm = deepcopy(gm_ori)

    while out_gm.n > L:
        c_ij = ise_cost(out_gm)

        merge_idx = np.unravel_index(np.argmin(c_ij), c_ij.shape)
        out_gm = merge_gm(out_gm, [merge_idx])

    return out_gm


def ise_cost(gm: GM):
    """Return the ise increase when two components of mixtures are merged

    Returns:
        np.ndarray: n by n matrix, (i, j) element is ise increase when i-th, j-th components are merged,
                    and diagonal terms are infinite

    """

    # merge
    pi_sum_ij = gm.pi[..., None] + gm.pi
    pi_prod_ij = gm.pi[..., None] * gm.pi
    _mu0 = gm.pi[..., None] * gm.mu
    mu_ij = (1. / pi_sum_ij)[..., None] * (np.expand_dims(_mu0, axis=1) + _mu0)
    _var0 = gm.pi[..., None, None] * gm.var
    _var1 = np.einsum(
        '...i,...j->...ij',
        np.expand_dims(gm.mu, axis=1) - gm.mu,
        np.expand_dims(gm.mu, axis=1) - gm.mu,
    )
    var_ij = 1. / pi_sum_ij[..., None, None] * (np.expand_dims(_var0, axis=1) + _var0) \
             + (pi_prod_ij / pi_sum_ij ** 2)[..., None, None] * _var1

    # four cost terms
    cost_cross = pi_prod_ij * integral_prod_gauss_prob(gm.mu, gm.var, gm.mu, gm.var, mode='cross')
    cost_self = np.diag(cost_cross)
    cost_merge = pi_sum_ij ** 2 * integral_prod_gauss_prob(mu_ij, var_ij, mu_ij, var_ij, mode='self')
    cost_trans = gm.pi[..., None] * pi_sum_ij * integral_prod_gauss_prob(
        np.expand_dims(gm.mu, axis=1),
        np.expand_dims(gm.var, axis=1),
        mu_ij,
        var_ij,
        mode='self',
    )

    cost = cost_self[..., None] + cost_self + 2 * cost_cross + cost_merge - 2 * (cost_trans + cost_trans.T)

    # make diagonal term infinite
    cost[np.arange(gm.n), np.arange(gm.n)] = np.inf * np.ones(gm.n)

    return cost
