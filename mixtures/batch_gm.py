from typing import List, Iterable

import numpy as np

from mixtures.gm import GM
from mixtures.utils import integral_prod_gauss_prob, prod_gauss_dist


class BatchGM(GM):
    """ Batch Gaussian Mixture """

    def __init__(
            self,
            pi: np.ndarray or list,     # components' weight, (b, n)
            mu: np.ndarray or list,     # components' mean, (b, n, d)
            var: np.ndarray or list,    # components' covariance matrix, (b, n, d, d)
    ):
        self.batch_form = True
        super().__init__(pi=pi, mu=mu, var=var)

    def __mul__(self, other: 'BatchGM') -> 'BatchGM':
        if not (self.b, self.n, self.d) == (other.b, self.n, self.d):
            ValueError(f"Two BatchGMs have different shape, "
                       f"BatchGM0: ({self.b}, {self.n}, {self.d}), BatchGM1: ({other.b}, {other.n}, {other.d})")

        _s = integral_prod_gauss_prob(self.mu, self.var, other.mu, other.var, mode='cross')
        _pi = np.reshape(_s * np.expand_dims(self.pi, axis=-1) * np.expand_dims(other.pi, axis=-2), (self.b, -1))
        _mu, _var = prod_gauss_dist(self.mu, self.var, other.mu, other.var, mode='cross')

        pi = _pi / np.sum(_pi, axis=-1, keepdims=True)
        mu = np.reshape(_mu, (self.b, -1, self.d))
        var = np.reshape(_var, (self.b, -1, self.d, self.d))

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
            np.random.seed(seed)

        gm_list = [GM.sample_gm(n, d, pi_alpha, mu_rng, var_df, var_scale) for _ in range(b)]
        pi = np.stack([gm.pi for gm in gm_list], axis=0)
        mu = np.stack([gm.mu for gm in gm_list], axis=0)
        var = np.stack([gm.var for gm in gm_list], axis=0)

        return BatchGM(pi, mu, var)

    def merge(self, idx_list: List[Iterable | np.ndarray]):
        """Batch-wise Merge each gaussian mixture in batch
        Only one round of two component merge is expected

        Args:
            idx_list: list of index which has same size with batch or array of (B, 2)

        """

        idx_list = np.array(idx_list)
        if idx_list.shape != (self.b, 2):
            ValueError('indices list have wrong dimensions')
        if not np.all(idx_list[:, 0] != idx_list[:, 1]):
            ValueError('overlapped indices are not allowed')

        target_pi = np.take_along_axis(self.pi, idx_list, axis=1)                       # B x 2
        target_mu = np.take_along_axis(self.mu, idx_list[..., None], axis=1)            # B x 2 x D
        target_var = np.take_along_axis(self.var, idx_list[..., None, None], axis=1)    # B x 2 x D x D

        _pi = np.sum(target_pi, axis=-1, keepdims=True)
        _mu = 1. / _pi[..., None] * np.sum(target_pi[..., None] * target_mu, axis=1, keepdims=True)
        _btw = np.einsum('...i,...j->...ij', target_mu - _mu, target_mu - _mu)
        _var = 1. / _pi[..., None, None] * np.sum(target_pi[..., None, None] * (target_var + _btw), axis=1, keepdims=True)

        ori_idx = np.array([np.setdiff1d(np.arange(self.n), idx) for idx in idx_list])
        self.n = self.n - 1
        self.pi = np.concatenate([np.take_along_axis(self.pi, ori_idx, axis=1), _pi], axis=1)
        self.mu = np.concatenate([np.take_along_axis(self.mu, ori_idx[..., None], axis=1), _mu], axis=1)
        self.var = np.concatenate([np.take_along_axis(self.var, ori_idx[..., None, None], axis=1), _var], axis=1)
