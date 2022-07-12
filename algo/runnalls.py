import numpy as np
from copy import deepcopy

from mixtures.gm import GM, merge_gm


def fit_runnalls(gm_ori: GM, L: int):
    """ find gaussian mixture of L component which is close to original one by Runnalls' algorithm

    Args:
        gm_ori: target gaussian mixture
        L: the number of components of fitted mixture

    Returns:
        GM: fitted gaussian mixture

    """

    out_gm = deepcopy(gm_ori)

    while out_gm.n > L:
        c_ij = runnalls_cost(out_gm)

        merge_idx = np.unravel_index(np.argmin(c_ij), c_ij.shape)
        out_gm = merge_gm(out_gm, [merge_idx])

    return out_gm


def runnalls_cost(gm: GM):
    """ calculate the upper bound of KL divergence increase when two components are merged

    Returns:
        np.ndarray: n by n matrix whose diagonal terms are infinite

    """

    pi_sum_ij = (gm.pi[..., None] + gm.pi)[..., None, None]
    pi_prod_ij = (gm.pi[..., None] * gm.pi)[..., None, None]
    term0 = gm.pi[..., None, None] * gm.var
    term1 = np.einsum(
        '...i,...j->...ij',
        np.expand_dims(gm.mu, axis=1) - gm.mu,
        np.expand_dims(gm.mu, axis=1) - gm.mu,
    )
    var_ij = 1. / pi_sum_ij * (np.expand_dims(term0, axis=1) + term0) \
             + pi_prod_ij / pi_sum_ij ** 2 * term1

    pi_det_ij = pi_sum_ij[..., 0, 0] * np.log(np.linalg.det(var_ij))
    pi_det_i = gm.pi * np.log(np.linalg.det(gm.var))
    c_ij = 0.5 * (pi_det_ij - pi_det_i[..., None] - pi_det_i)

    # make diagonal term infinite
    c_ij[np.arange(gm.n), np.arange(gm.n)] = np.inf * np.ones(gm.n)

    return c_ij
