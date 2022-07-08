import numpy as np
from copy import deepcopy

from gm import GM, merge_gm


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
        np.ndarray: n by n matrix

    """

    pi_ij = gm.pi + gm.pi[..., None]
    mu_inner = gm.pi * gm.mu
    mu_ij = 1. / pi_ij * (mu_inner + mu_inner[..., None])
    var_inner0 = gm.pi * gm.var
    var_inner1 = gm.pi[..., None] * (gm.mu - mu_ij) ** 2
    var_ij = 1. / pi_ij * (var_inner0 + var_inner0[..., None] + var_inner1 + var_inner1.T)

    pi_logvar = np.pi * np.log(gm.var)
    c_ij = 0.5 * (pi_ij * np.log(var_ij) - pi_logvar - pi_logvar[..., None])

    # make diagonal term infinite
    c_ij[np.arange(gm.n), np.arange(gm.n)] = np.inf * np.ones(gm.n)

    return c_ij
