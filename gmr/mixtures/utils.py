import math
from typing import Tuple

import torch


def setdiff1d(tensor0: torch.Tensor, tensor1: torch.Tensor):
    """Find the set difference of two vector tensors.
    Return the unique values in tensor0 that are not in tensor1.

    Args:
        tensor0: tensor of (N,)
        tensor1: tensor of (M,)

    Returns:
        torch.Tensor: tensor of (..., L) where L <= N

    """

    notin_idx = torch.ne(tensor0.unsqueeze(dim=-1), tensor1.unsqueeze(dim=-2)).all(dim=-1)

    return tensor0[notin_idx]


def gauss_prob(x: torch.Tensor, mu: torch.Tensor, var: torch.Tensor) -> torch.Tensor:
    """Return probability of x for (mu, var) distributed Gaussian

    Args:
        x: tensor of (..., D)
        mu: tensor of ([B], N, D)
        var: tensor of ([B], N, D, D)

    Returns:
        torch.Tensor: tensor of (..., [B], N)

    """

    d = x.shape[-1]
    ex_x = x.view(x.shape[:-1] + (1,) * (mu.ndim - 1) + (x.shape[-1],))
    ex_var = torch.broadcast_to(var, x.shape[:-1] + var.shape)

    term0 = d * math.log(2 * math.pi) + torch.log(torch.linalg.det(var))
    term1 = torch.einsum('...i,...i->...', ex_x - mu, torch.linalg.solve(ex_var, ex_x - mu))
    prob = torch.exp(- 0.5 * (term0 + term1))

    return prob


def integral_prod_gauss_prob(mu0: torch.Tensor, var0: torch.Tensor, mu1: torch.Tensor, var1: torch.Tensor,
                             mode='self') -> torch.Tensor:
    """Return integration of product of two gaussian

    Args:
        mu0: tensor of (..., N, D)
        var0: tensor of (..., N, D, D)
        mu1: tensor of (..., M, D)
        var1: tensor of (..., M, D, D)
        mode: 'self' or 'cross'

    Returns:
        torch.Tensor: tensor of (..., N) if mode is 'self', (..., N, M) otherwise

    """

    d = mu0.size(-1)

    if mode == 'self':
        diff_mu_ij = mu0 - mu1                                                      # (..., N, D)
        sum_var_ij = var0 + var1                                                    # (..., N, D, D)
    elif mode == 'cross':
        diff_mu_ij = mu0.unsqueeze(dim=-2) - mu1.unsqueeze(dim=-3)                  # (..., N, M, D)
        sum_var_ij = var0.unsqueeze(dim=-3) + var1.unsqueeze(dim=-4)                # (..., N, M, D, D)
    else:
        raise ValueError(f"mode(:{mode}) should be in ['self', 'cross'].")

    term0 = d * math.log(2 * math.pi) + torch.log(torch.linalg.det(sum_var_ij))
    term1 = torch.einsum('...i,...i->...', diff_mu_ij, torch.linalg.solve(sum_var_ij, diff_mu_ij))
    prob = torch.exp(- 0.5 * (term0 + term1))

    return prob


def prod_gauss_dist(mu0: torch.Tensor, var0: torch.Tensor, mu1: torch.Tensor, var1: torch.Tensor,
                    mode='self') -> Tuple[torch.Tensor, torch.Tensor]:
    """Return gaussian parameters which is proportional to the product of two gaussian

    Args:
        mu0: tensor of (..., N, D)
        var0: tensor of (..., N, D, D)
        mu1: tensor of (..., M, D)
        var1: tensor of (..., M, D, D)
        mode: 'self' or 'cross'

    Returns:
        torch.Tensor: tensor of (..., N, D) if mode is 'self', (..., N, M, D) otherwise
        torch.Tensor: tensor of (..., N, D, D) if mode is 'self', (..., N, M, D, D) otherwise

    """

    _inv_var0 = torch.linalg.inv(var0)
    _inv_var1 = torch.linalg.inv(var1)
    _inv_var_mu0 = torch.linalg.solve(var0, mu0)
    _inv_var_mu1 = torch.linalg.solve(var1, mu1)

    if mode == 'self':
        _sum_inv_var = _inv_var0 + _inv_var1
        _sum_inv_var_mul_mu = _inv_var_mu0 + _inv_var_mu1
    elif mode == 'cross':
        _sum_inv_var = _inv_var0.unsqueeze(dim=-3) + _inv_var1.unsqueeze(dim=-4)
        _sum_inv_var_mul_mu = _inv_var_mu0.unsqueeze(dim=-2) + _inv_var_mu1.unsqueeze(dim=-3)
    else:
        raise ValueError(f"mode(:{mode}) should be in ['self', 'cross'].")

    prod_var = torch.linalg.inv(_sum_inv_var)
    prod_mu = torch.einsum('...ij,...j->...i', prod_var, _sum_inv_var_mul_mu)

    return prod_mu, prod_var
