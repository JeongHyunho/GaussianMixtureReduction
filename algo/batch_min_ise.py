import numpy as np
from copy import deepcopy

from mixtures.batch_gm import BatchGM, batch_merge_gm
from mixtures.utils import integral_prod_gauss_prob


def fit_batch_min_ise(batch_gm_ori: BatchGM, L: int):
    """Batch version of min-ise algorithm
    It iteratively merge two components of mixture which incurs minimum ise increase

    Args:
        batch_gm_ori: the gaussian mixture to be reduced
        L: the number of components of reduced mixture

    Returns:
        BatchGM: reduced gaussian mixture

    """

    out_batch_gm = deepcopy(batch_gm_ori)

    while out_batch_gm.n > L:
        c_bij = batch_ise_cost(out_batch_gm)

        merge_idx = [np.unravel_index(np.argmin(c_ij), c_ij.shape) for c_ij in c_bij]
        out_batch_gm = batch_merge_gm(out_batch_gm, merge_idx)

    return out_batch_gm


def batch_ise_cost(batch_gm: BatchGM):
    """Return the batch-wise ise increase when two components of mixtures are merged

    Returns:
        np.ndarray: (b, n, n) matrix, (i, j, k) element is ise increase when j-th, k-th components in i-th batch
                are merged, and for each batch diagonal terms are infinite

    """

    # merge
    pi_sum_ij = batch_gm.pi[..., None] + np.expand_dims(batch_gm.pi, axis=1)
    pi_prod_ij = batch_gm.pi[..., None] * np.expand_dims(batch_gm.pi, axis=1)
    _mu0 = batch_gm.pi[..., None] * batch_gm.mu
    mu_ij = (1. / pi_sum_ij)[..., None] * (np.expand_dims(_mu0, axis=2) + np.expand_dims(_mu0, axis=1))
    _var0 = batch_gm.pi[..., None, None] * batch_gm.var
    _var1 = np.expand_dims(batch_gm.mu, axis=2) - np.expand_dims(batch_gm.mu, axis=1)
    _var2 = np.einsum('...i,...j->...ij', _var1, _var1)
    var_ij = 1. / pi_sum_ij[..., None, None] * (np.expand_dims(_var0, axis=2) + np.expand_dims(_var0, axis=1)) \
             + (pi_prod_ij / pi_sum_ij ** 2)[..., None, None] * _var2

    # four cost terms
    cost_cross = pi_prod_ij * integral_prod_gauss_prob(batch_gm.mu, batch_gm.var, batch_gm.mu, batch_gm.var, mode='cross')
    cost_self = cost_cross[..., np.arange(batch_gm.n), np.arange(batch_gm.n)]
    cost_merge = pi_sum_ij ** 2 * integral_prod_gauss_prob(mu_ij, var_ij, mu_ij, var_ij, mode='self')
    cost_trans = batch_gm.pi[..., None] * pi_sum_ij * integral_prod_gauss_prob(
        np.expand_dims(batch_gm.mu, axis=2),
        np.expand_dims(batch_gm.var, axis=2),
        mu_ij,
        var_ij,
        mode='self',
    )

    cost = np.expand_dims(cost_self, axis=2) + np.expand_dims(cost_self, axis=1) \
           + 2 * cost_cross + cost_merge - 2 * (cost_trans + np.swapaxes(cost_trans, axis1=1, axis2=2))

    # make diagonal term infinite
    cost[..., np.arange(batch_gm.n), np.arange(batch_gm.n)] = np.inf * np.ones(batch_gm.n)

    return cost
