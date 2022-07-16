from __future__ import annotations

import math
from typing import List, Iterable, Tuple

import torch
from torch import nn
from torch.distributions import dirichlet, wishart

from .helpers import check_dim, check_batch
from .utils import gauss_prob, integral_prod_gauss_prob, prod_gauss_dist, setdiff1d


class Options(object):
    _instance = None
    device = 'cpu'

    def __new__(cls):
        if cls._instance is None:
            cls._instance = object.__new__(cls)
        return cls._instance

    def __repr__(self):
        return "Options: [device: %s]" % self.device


options = Options()


class GM(nn.Module):
    """ Gaussian Mixture """

    batch_form: bool = False

    def __init__(
            self,
            pi: torch.Tensor | Tuple[torch.Tensor] | List[torch.Tensor],     # components' weight, (n,)
            mu: torch.Tensor | Tuple[torch.Tensor] | List[torch.Tensor],     # components' mean, (n, d)
            var: torch.Tensor | Tuple[torch.Tensor] | List[torch.Tensor],    # components' covariance matrix, (n, d, d)
    ):
        super().__init__()

        # non-ndarray handling
        self.register_buffer('pi', pi if isinstance(pi, torch.Tensor) else torch.tensor(pi))
        self.register_buffer('mu', mu if isinstance(mu, torch.Tensor) else torch.stack(mu, dim=0))
        self.register_buffer('var', var if isinstance(var, torch.Tensor) else torch.stack(var, dim=0))
        self.to(options.device)

        # check dimension
        self.n, self.d = check_dim(self.pi, self.mu, self.var)  # the number of mixture components, feature dim

        # the size of batch
        self.b = check_batch(self.pi, self.mu, self.var, self.batch_form)

    def __repr__(self):
        return f"b:{self.b}\nn:{self.n}\nd:{self.d}\npi:\n{self.pi}\nmu:\n{self.mu}\nvar:\n{self.var}"

    def __eq__(self, other: 'GM'):
        if self.n != other.n:
            return False
        elif torch.any(torch.gt(torch.abs(self.mu - other.mu), 1e-5)):
            return False
        elif torch.any(torch.gt(torch.abs(self.var - other.var), 1e-5)):
            return False
        else:
            return True

    def __mul__(self, other: 'GM') -> 'GM':
        if not (self.n, self.d) == (other.n, other.d):
            ValueError(f"Two GMs have different shape, "
                       f"GM0: ({self.b}, {self.n}, {self.d}), GM1: ({other.b}, {other.n}, {other.d})")

        _s = integral_prod_gauss_prob(self.mu, self.var, other.mu, other.var, mode='cross')
        _pi = torch.ravel(_s * self.pi[..., None] * other.pi)
        _mu, _var = prod_gauss_dist(self.mu, self.var, other.mu, other.var, mode='cross')

        pi = _pi / torch.sum(_pi)
        mu = _mu.view(-1, self.d)
        var = _var.view(-1, self.d, self.d)

        return GM(pi, mu, var)

    @staticmethod
    def sample_gm(n, d, pi_alpha, mu_rng, var_df, var_scale, seed=None):
        """Sample specified gaussian mixture, mean from uniform, var from Wishart distribution

         Returns:
             GM: sampled mixture

         """

        assert len(mu_rng) == 2 and mu_rng[0] <= mu_rng[1], f'mu_rng of [min, max] is expected, but {mu_rng}'

        if seed is not None:
            torch.manual_seed(seed)

        pi = dirichlet.Dirichlet(pi_alpha).sample()
        mu = torch.stack([torch.rand(d) * (mu_rng[1] - mu_rng[0]) + mu_rng[0] for _ in range(n)], dim=0)
        wishart_dist = wishart.Wishart(df=var_df, covariance_matrix=var_scale)
        var = torch.stack([wishart_dist.sample() for _ in range(n)], dim=0)
        var = var[..., None, None] if d == 1 else var
        out_gm = GM(pi=pi, mu=mu, var=var)

        return out_gm

    def prob(self, t: torch.Tensor, reduce=True):
        """Return gaussian mixture's probability on t

        Args:
            t: tensor of (..., D) and D is feature dimension of the mixture
            reduce: if True, do sum reduction on last dim

        Returns:
            torch.Tensor: tensor of (..., [B]) if 'reduce' is True, (..., [B], D) otherwise

        """

        if t.size(-1) != self.d:
            raise ValueError(f'query points have different feature dim ({t.shape[-1]}), not {self.d}')

        _pi = self.pi if reduce else self.pi[..., None]
        prob = torch.sum(_pi * gauss_prob(t, self.mu, self.var, reduce=reduce), dim=-1 if reduce else -2)

        return prob

    def merge(self, idx_list: Iterable):
        """Merge indexed gaussian mixture components with preserved moments

        Args:
            idx_list: list of index for merged components

        """

        idx_list = [torch.tensor(idx, device=self.pi.device).long() for idx in idx_list]
        flatten_idx = torch.cat(idx_list, dim=-1)
        assert len(flatten_idx) == len(set(flatten_idx)), 'Overlapped indices of components to be merged'

        merge_pi = []
        merge_mu = []
        merge_var = []

        for idx in idx_list:
            target_pi = torch.take(self.pi, idx)
            target_mu = self.mu[idx]
            target_var = self.var[idx]

            _pi = torch.sum(target_pi)
            _mu = 1. / _pi * torch.sum(target_pi[..., None] * target_mu, dim=0)
            _btw = torch.einsum('...i, ...j -> ...ij', target_mu - _mu, target_mu - _mu)
            _var = 1. / _pi * torch.sum(target_pi[..., None, None] * (target_var + _btw), dim=0)

            merge_pi.append(_pi)
            merge_mu.append(_mu)
            merge_var.append(_var)

        ori_i = setdiff1d(torch.arange(self.n).to(flatten_idx), flatten_idx)
        self.n = len(ori_i) + 1
        self.pi = torch.cat([self.pi[ori_i], torch.stack(merge_pi, dim=0)], dim=-1).type_as(self.pi)
        self.mu = torch.cat([self.mu[ori_i], torch.stack(merge_mu, dim=0)], dim=0).type_as(self.mu)
        self.var = torch.cat([self.var[ori_i], torch.stack(merge_var, dim=0)], dim=0).type_as(self.var)


def calc_ise(gm0: GM, gm1: GM) -> torch.Tensor:
    """Integrated square error of two gaussian mixtures

    Returns:
        float: difference of two gm, scalar number

    """

    H0 = calc_integral_outer_prod_gm(gm0, gm0)
    H1 = calc_integral_outer_prod_gm(gm0, gm1)
    H2 = calc_integral_outer_prod_gm(gm1, gm1)

    J00 = gm0.pi @ H0 @ gm0.pi
    J01 = gm0.pi @ H1 @ gm1.pi
    J11 = gm1.pi @ H2 @ gm1.pi

    ise = J00 - 2 * J01 + J11

    return ise


def calc_integral_outer_prod_gm(gm0: GM, gm1: GM):
    """Integrates the outer product of two gaussian mixtures

    Returns:
        np.ndarray: n by n matrix

    """

    assert gm0.d == gm1.d, 'different dims'
    d = gm0.d

    x_i = gm0.mu.unsqueeze(dim=1)
    mu_j = gm1.mu
    var_ij = gm0.var.unsqueeze(dim=1) + gm1.var

    term0 = d * math.log(2 * math.pi) + torch.log(torch.linalg.det(var_ij))
    term1 = torch.einsum('...i,...i->...', x_i - mu_j, torch.linalg.solve(var_ij, x_i - mu_j))
    H = torch.exp(- 0.5 * (term0 + term1))

    return H


def kl_gm_comp(gm0: GM, gm1: GM):
    """Calculates KL divergence between all components in two gaussian mixtures

    Args:
        gm0: mixture which has N components
        gm1: mixture which has M components

    Returns:
        np.ndarray: N by M matrix, (i, j) element is kl btw i-th and j-th components from gm0 and gm1 respectively

    """

    assert gm0.d == gm1.d, 'different dims'
    d = gm0.d

    _mu_diff_ij = gm0.mu.unsqueeze(dim=1) - gm1.mu
    _btw_var_ij = torch.einsum('...i, ...j -> ...ij', _mu_diff_ij, _mu_diff_ij)
    _sum_var_ij = gm0.var.unsqueeze(dim=1) + _btw_var_ij
    _trace_term = torch.einsum('...ii->...', torch.linalg.solve(gm1.var, _sum_var_ij) - torch.eye(d).type_as(gm1.var))
    _log_det_ratio = torch.log(torch.linalg.det(gm1.var) / torch.linalg.det(gm0.var).unsqueeze(dim=-1))

    kl = 0.5 * (_trace_term + _log_det_ratio)
    return kl
