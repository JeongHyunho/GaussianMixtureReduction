import torch
import numpy as np
from copy import deepcopy

from gmr.mixtures.gm import GM


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

        merge_idx = np.unravel_index(torch.argmin(c_ij).cpu(), c_ij.shape)
        out_gm.merge([merge_idx])

    return out_gm


def runnalls_cost(gm: GM):
    """ calculate the upper bound of KL divergence increase when two components are merged

    Returns:
        torch.Tensor: n by n matrix whose diagonal terms are infinite

    """

    pi_sum_ij = (gm.pi[..., None] + gm.pi)[..., None, None]
    pi_prod_ij = (gm.pi[..., None] * gm.pi)[..., None, None]
    term0 = gm.pi[..., None, None] * gm.var
    mu_diff = gm.mu.unsqueeze(dim=1) - gm.mu
    term1 = torch.einsum('...i,...j->...ij', mu_diff, mu_diff)
    var_ij = 1. / pi_sum_ij * (term0.unsqueeze(dim=1) + term0) \
             + pi_prod_ij / pi_sum_ij ** 2 * term1

    pi_det_ij = pi_sum_ij[..., 0, 0] * torch.log(torch.linalg.det(var_ij))
    pi_det_i = gm.pi * torch.log(torch.linalg.det(gm.var))
    c_ij = 0.5 * (pi_det_ij - pi_det_i[..., None] - pi_det_i)

    # make diagonal term infinite
    c_ij[torch.arange(gm.n), torch.arange(gm.n)] = torch.inf * torch.ones(gm.n).type_as(c_ij)

    return c_ij
