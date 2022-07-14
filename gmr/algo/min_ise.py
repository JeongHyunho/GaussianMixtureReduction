from __future__ import annotations

import torch
import numpy as np
from copy import deepcopy

from gmr.mixtures.gm import GM
from gmr.mixtures.batch_gm import BatchGM
from gmr.mixtures.utils import integral_prod_gauss_prob


def fit_min_ise(gm_ori: GM | BatchGM, L: int, inplace=False):
    """Iteratively merge two components of mixture which incurs minimum ise increase

    Args:
        gm_ori: the (batch of) gaussian mixture to be reduced
        L: the number of components of reduced mixture
        inplace: change mixture in arg or not

    Returns:
        GM: reduced (batch of) gaussian mixture

    """

    assert gm_ori.n > L, 'More components in the mixture than {L} are required'

    if inplace:
        out_gm = gm_ori
    else:
        out_gm = deepcopy(gm_ori)

    while out_gm.n > L:
        cost = ise_cost(out_gm)

        if type(gm_ori) is GM:
            merge_idx = np.unravel_index(torch.argmin(cost).cpu(), cost.shape)
            out_gm.merge([merge_idx])
        else:
            merge_idx = [np.unravel_index(torch.argmin(c_ij).cpu(), c_ij.shape) for c_ij in cost]
            out_gm.merge(merge_idx)

    return out_gm


def ise_cost(gm: GM | BatchGM):
    """Return the ise increase when two components of mixtures are merged

    Returns:
        torch.Tensor: ([b], n, n) array, ([i], j, k) element is ise increase when j-th, k-th components [in i-th batch]
                    are merged, and diagonal terms are infinite

    """

    # merge
    pi_sum_ij = gm.pi.unsqueeze(dim=-1) + gm.pi.unsqueeze(dim=-2)
    pi_prod_ij = gm.pi.unsqueeze(dim=-1) * gm.pi.unsqueeze(dim=-2)                                      # (..., N, N)
    _mu0 = gm.pi.unsqueeze(dim=-1) * gm.mu                                                              # (..., N, D)
    mu_ij = (1. / pi_sum_ij)[..., None] * (_mu0.unsqueeze(dim=-2) + _mu0.unsqueeze(dim=-3))             # (..., N, N, D)
    _var0 = gm.pi[..., None, None] * gm.var                                                             # (..., N, D, D)
    _mu_diff_ij = gm.mu.unsqueeze(dim=-2) - gm.mu.unsqueeze(dim=-3)                                     # (..., N, N, D)
    _var_btw_ij = torch.einsum('...i,...j->...ij', _mu_diff_ij, _mu_diff_ij)
    var_ij = 1. / pi_sum_ij[..., None, None] * (_var0.unsqueeze(dim=-3) + _var0.unsqueeze(dim=-4)) \
             + (pi_prod_ij / pi_sum_ij ** 2)[..., None, None] * _var_btw_ij                          # (..., N, N, D, D)

    # four cost terms
    cost_cross = pi_prod_ij * integral_prod_gauss_prob(gm.mu, gm.var, gm.mu, gm.var, mode='cross')
    cost_self = torch.einsum('...ii->...i', cost_cross)
    cost_merge = pi_sum_ij ** 2 * integral_prod_gauss_prob(mu_ij, var_ij, mu_ij, var_ij, mode='self')
    cost_trans = gm.pi[..., None] * pi_sum_ij * integral_prod_gauss_prob(
        gm.mu.unsqueeze(dim=-2), gm.var.unsqueeze(dim=-3), mu_ij, var_ij, mode='self')

    cost = cost_self.unsqueeze(dim=-1) + cost_self.unsqueeze(dim=-2) \
           + 2 * cost_cross + cost_merge - 2 * (cost_trans + torch.transpose(cost_trans, dim0=-1, dim1=-2))

    # make diagonal term infinite
    cost[..., torch.arange(gm.n), torch.arange(gm.n)] = torch.inf * torch.ones(gm.n).type_as(cost)

    return cost
