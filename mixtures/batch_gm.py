from dataclasses import dataclass
from typing import List, Iterable

import numpy as np

from mixtures.gm import GM


@dataclass
class BatchGM(GM):
    """ Batch Gaussian Mixture """

    def __post_init__(self):
        self.batch_form = True
        super().__post_init__()

    def __mul__(self, other):
        raise NotImplemented

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


def batch_merge_gm(batch_gm: BatchGM, idx_list: List[Iterable | np.ndarray]):
    """Batch-wise Merge each gaussian mixture in batch
    Only one round of two component merge is expected

    Args:
        batch_gm: batch of mixtures to be merged
        idx_list: list of index which has same size with batch or array of (B, 2)

    Returns:
        BatchGM: merged batch mixture

    """

    idx_list = np.array(idx_list)
    assert idx_list.shape == (batch_gm.b, 2), ValueError('indices list have wrong dimensions')
    assert np.all(idx_list[:, 0] != idx_list[:, 1]), ValueError('overlapped indices are not allowed')

    target_pi = np.take_along_axis(batch_gm.pi, idx_list, axis=1)                       # B x 2
    target_mu = np.take_along_axis(batch_gm.mu, idx_list[..., None], axis=1)            # B x 2 x D
    target_var = np.take_along_axis(batch_gm.var, idx_list[..., None, None], axis=1)    # B x 2 x D x D

    _pi = np.sum(target_pi, axis=-1, keepdims=True)
    _mu = 1. / _pi[..., None] * np.sum(target_pi[..., None] * target_mu, axis=1, keepdims=True)
    _btw = np.einsum('...i,...j->...ij', target_mu - _mu, target_mu - _mu)
    _var = 1. / _pi[..., None, None] * np.sum(target_pi[..., None, None] * (target_var + _btw), axis=1, keepdims=True)

    ori_idx = np.array([np.setdiff1d(np.arange(batch_gm.n), idx) for idx in idx_list])
    pi = np.concatenate([np.take_along_axis(batch_gm.pi, ori_idx, axis=1), _pi], axis=1)
    mu = np.concatenate([np.take_along_axis(batch_gm.mu, ori_idx[..., None], axis=1), _mu], axis=1)
    var = np.concatenate([np.take_along_axis(batch_gm.var, ori_idx[..., None, None], axis=1), _var], axis=1)

    out_batch_gm = BatchGM(pi=pi, mu=mu, var=var)
    return out_batch_gm
