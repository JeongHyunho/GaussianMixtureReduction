from __future__ import annotations

from typing import List

import torch
from torch.distributions import dirichlet, wishart

from .gm import GM
from .helpers import check_batch
from .utils import integral_prod_gauss_prob, prod_gauss_dist, setdiff1d, clamp_inf


class BatchGM(GM):
    """ Batch Gaussian Mixture """

    def __init__(self, *args, **kwargs):
        self.batch_form = True
        super().__init__(*args, **kwargs)

    def __mul__(self, other: 'BatchGM') -> 'BatchGM':
        if not (*self.b, self.n, self.d) == (*other.b, other.n, other.d):
            ValueError(f"Two BatchGMs have different shape, "
                       f"BatchGM0: (*{self.b}, {self.n}, {self.d}), BatchGM1: (*{other.b}, {other.n}, {other.d})")

        _s = integral_prod_gauss_prob(self.mu, self.var, other.mu, other.var, mode='cross')
        _s = clamp_inf(_s)
        _pi = (_s * self.pi.unsqueeze(dim=-1) * other.pi.unsqueeze(dim=-2)).view(*self.b, -1)
        _mu, _var = prod_gauss_dist(self.mu, self.var, other.mu, other.var, mode='cross')

        pi = _pi / torch.sum(_pi, dim=-1, keepdim=True)
        mu = _mu.view(*self.b, -1, self.d)
        var = _var.view(*self.b, -1, self.d, self.d)

        return BatchGM(pi=pi, mu=mu, var=var)

    def __getitem__(self, item):
        pi = self.pi[item]
        mu = self.mu[item]
        var = self.var[item]

        if pi.ndim > 1:
            return BatchGM(pi, mu, var)
        else:
            return GM(pi, mu, var)

    def to_gm(self) -> List[GM]:
        return [GM(pi=_pi, mu=_mu, var=_var) for _pi, _mu, _var in zip(self.pi, self.mu, self.var)]

    @staticmethod
    def sample_batch_gm(n, b, d, pi_alpha, mu_rng, var_df, var_scale, seed=None):
        """Sample batch of gaussian mixtures

        Returns:
            BatchGM: sampled mixture batch

        """

        assert len(mu_rng) == 2 and mu_rng[0] <= mu_rng[1], f'mu_rng of [min, max] is expected, but {mu_rng}'

        if not hasattr(b, '__iter__'):
            b = [b]

        if seed is not None:
            torch.manual_seed(seed)

        pi = dirichlet.Dirichlet(pi_alpha).sample(b)
        mu = torch.rand(b + [n, d]) * (mu_rng[1] - mu_rng[0]) + mu_rng[0]
        var = wishart.Wishart(df=var_df, covariance_matrix=var_scale).sample(b + [n])

        return BatchGM(pi, mu, var)

    def merge(self, idx_list: torch.Tensor | List[torch.Tensor]):
        """Batch-wise Merge each gaussian mixture in batch
        Only one round of two component merge is expected

        Args:
            idx_list: list of index which has same size with batch or array of (*B, 2)

        """

        idx_list = torch.tensor(idx_list, device=self.pi.device).long()
        if idx_list.shape != (*self.b, 2):
            ValueError('indices list have wrong dimensions')
        if torch.any(torch.eq(idx_list[..., 0], idx_list[..., 1])):
            ValueError('overlapped indices are not allowed')

        target_pi = torch.take_along_dim(self.pi, idx_list, dim=-1)                       # *B x 2
        target_mu = torch.take_along_dim(self.mu, idx_list[..., None], dim=-2)            # *B x 2 x D
        target_var = torch.take_along_dim(self.var, idx_list[..., None, None], dim=-3)    # *B x 2 x D x D

        _pi = torch.sum(target_pi, dim=-1, keepdim=True)
        _mu = 1. / (_pi[..., None] + 1e-6) * \
              torch.sum(target_pi[..., None] * target_mu, dim=-2, keepdim=True)
        _btw = torch.einsum('...i,...j->...ij', target_mu - _mu, target_mu - _mu)
        _var = 1. / (_pi[..., None, None] + 1e-6) * \
               torch.sum(target_pi[..., None, None] * (target_var + _btw), dim=-3, keepdim=True)

        all_idx = torch.broadcast_to(torch.arange(self.n), (*self.b, self.n)).to(idx_list)
        ori_idx = setdiff1d(all_idx, idx_list)
        self.n = self.n - 1
        self.pi = torch.cat([torch.take_along_dim(self.pi, ori_idx, dim=-1), _pi], dim=-1)
        self.mu = torch.cat([torch.take_along_dim(self.mu, ori_idx[..., None], dim=-2), _mu], dim=-2)
        self.var = torch.cat([torch.take_along_dim(self.var, ori_idx[..., None, None], dim=-3), _var], dim=-3)


def cat_bgm(bgm0: BatchGM, bgm1: BatchGM, dim=-1) -> BatchGM:
    """concatenate two gaussian mixtures along specified dim

    Args:
        bgm0: batch of mixture
        bgm1: batch of mixture
        dim: concatenation dimension

    Returns:
        BatchGM: concatenated batch of mixtures

    """

    bgm0.pi = torch.cat([bgm0.pi, bgm1.pi], dim=dim)
    bgm0.mu = torch.cat([bgm0.mu, bgm1.mu], dim=dim if dim >= 0 else dim - 1)
    bgm0.var = torch.cat([bgm0.var, bgm1.var], dim=dim if dim >= 0 else dim - 2)
    bgm0.b = check_batch(bgm0.pi, bgm0.mu, bgm0.var, batch_form=True)

    return bgm0
