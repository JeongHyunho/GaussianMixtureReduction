from typing import List

import torch

from gmr.mixtures.gm import GM
from gmr.mixtures.utils import integral_prod_gauss_prob, prod_gauss_dist, setdiff1d


class BatchGM(GM):
    """ Batch Gaussian Mixture """

    def __init__(self, *args, **kwargs):
        self.batch_form = True
        super().__init__(*args, **kwargs)

    def __mul__(self, other: 'BatchGM') -> 'BatchGM':
        if not (self.b, self.n, self.d) == (other.b, other.n, other.d):
            ValueError(f"Two BatchGMs have different shape, "
                       f"BatchGM0: ({self.b}, {self.n}, {self.d}), BatchGM1: ({other.b}, {other.n}, {other.d})")

        _s = integral_prod_gauss_prob(self.mu, self.var, other.mu, other.var, mode='cross')
        _pi = (_s * self.pi.unsqueeze(dim=-1) * other.pi.unsqueeze(dim=-2)).view(self.b, -1)
        _mu, _var = prod_gauss_dist(self.mu, self.var, other.mu, other.var, mode='cross')

        pi = _pi / torch.sum(_pi, dim=-1, keepdim=True)
        mu = _mu.view(self.b, -1, self.d)
        var = _var.view(self.b, -1, self.d, self.d)

        return BatchGM(pi=pi, mu=mu, var=var)

    def __getitem__(self, item):
        pi = self.pi[item]
        mu = self.mu[item]
        var = self.var[item]

        if isinstance(item, slice):
            return BatchGM(pi, mu, var)
        else:
            return GM(pi, mu, var)

    @staticmethod
    def sample_batch_gm(n, b, d, pi_alpha, mu_rng, var_df, var_scale, seed=None):
        """Sample batch of gaussian mixtures

        Returns:
            BatchGM: sampled mixture batch

        """

        if seed is not None:
            torch.manual_seed(seed)

        gm_list = [GM.sample_gm(n, d, pi_alpha, mu_rng, var_df, var_scale) for _ in range(b)]
        pi = torch.stack([gm.pi for gm in gm_list], dim=0)
        mu = torch.stack([gm.mu for gm in gm_list], dim=0)
        var = torch.stack([gm.var for gm in gm_list], dim=0)

        return BatchGM(pi, mu, var)

    def merge(self, idx_list: torch.Tensor | List[torch.Tensor]):
        """Batch-wise Merge each gaussian mixture in batch
        Only one round of two component merge is expected

        Args:
            idx_list: list of index which has same size with batch or array of (B, 2)

        """

        idx_list = torch.tensor(idx_list, device=self.pi.device).long()
        if idx_list.shape != (self.b, 2):
            ValueError('indices list have wrong dimensions')
        if torch.any(torch.eq(idx_list[:, 0], idx_list[:, 1])):
            ValueError('overlapped indices are not allowed')

        target_pi = torch.take_along_dim(self.pi, idx_list, dim=1)                       # B x 2
        target_mu = torch.take_along_dim(self.mu, idx_list[..., None], dim=1)            # B x 2 x D
        target_var = torch.take_along_dim(self.var, idx_list[..., None, None], dim=1)    # B x 2 x D x D

        _pi = torch.sum(target_pi, dim=-1, keepdim=True)
        _mu = 1. / _pi[..., None] * torch.sum(target_pi[..., None] * target_mu, dim=1, keepdim=True)
        _btw = torch.einsum('...i,...j->...ij', target_mu - _mu, target_mu - _mu)
        _var = 1. / _pi[..., None, None] * \
               torch.sum(target_pi[..., None, None] * (target_var + _btw), dim=1, keepdim=True)

        ori_idx = torch.stack([setdiff1d(torch.arange(self.n).to(idx), idx) for idx in idx_list], dim=0)
        self.n = self.n - 1
        self.pi = torch.cat([torch.take_along_dim(self.pi, ori_idx, dim=1), _pi], dim=1)
        self.mu = torch.cat([torch.take_along_dim(self.mu, ori_idx[..., None], dim=1), _mu], dim=1)
        self.var = torch.cat([torch.take_along_dim(self.var, ori_idx[..., None, None], dim=1), _var], dim=1)
