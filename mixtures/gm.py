from typing import List, Iterable

import numpy as np
from scipy.stats import wishart

from mixtures import check_var, check_dim, check_batch
from mixtures.utils import gauss_prob, integral_prod_gauss_prob, prod_gauss_dist


class GM:
    """ Gaussian Mixture """

    batch_form: bool = False

    def __init__(
            self,
            pi: np.ndarray or list,  # components' weight, (n,)
            mu: np.ndarray or list,  # components' mean, (n, d)
            var: np.ndarray or list,  # components' covariance matrix, (n, d, d)
    ):
        # non-ndarray handling
        self.pi = pi if isinstance(pi, np.ndarray) else np.array(pi)
        self.mu = mu if isinstance(mu, np.ndarray) else np.array(mu)
        self.var = var if isinstance(var, np.ndarray) else np.array(var)

        # check args
        self.n, self.d = check_dim(self.pi, self.mu, self.var)  # the number of mixture components, feature dim
        check_var(self.var)

        # the size of batch
        self.b = check_batch(self.pi, self.mu, self.var, self.batch_form)

    def __repr__(self):
        return f"b:{self.b}\nn:{self.n}\nd:{self.d}\npi:\n{self.pi}\nmu:\n{self.mu}\nvar:\n{self.var}"

    def __eq__(self, other: 'GM'):
        if self.n != other.n:
            return False
        elif np.any(np.abs(self.mu - other.mu) > 1e-9):
            return False
        elif np.any(np.abs(self.var - other.var) > 1e-9):
            return False
        else:
            return True

    def __mul__(self, other: 'GM') -> 'GM':
        if not (self.n, self.d) == (self.n, self.d):
            ValueError(f"Two GMs have different shape, "
                       f"GM0: ({self.b}, {self.n}, {self.d}), GM1: ({other.b}, {other.n}, {other.d})")

        _s = integral_prod_gauss_prob(self.mu, self.var, other.mu, other.var, mode='cross')
        _pi = np.ravel(_s * self.pi[..., None] * other.pi)
        _mu, _var = prod_gauss_dist(self.mu, self.var, other.mu, other.var, mode='cross')

        pi = _pi / np.sum(_pi)
        mu = np.reshape(_mu, (-1, self.d))
        var = np.reshape(_var, (-1, self.d, self.d))

        return GM(pi, mu, var)

    @staticmethod
    def sample_gm(n, d, pi_alpha, mu_rng, var_df, var_scale, seed=None):
        """Sample specified gaussian mixture, mean from uniform, var from Wishart distribution

         Returns:
             GM: sampled mixture

         """

        assert len(mu_rng) == 2 and mu_rng[0] <= mu_rng[1], f'mu_rng of [min, max] is expected, but {mu_rng}'

        if seed is not None:
            np.random.seed(seed)

        pi = np.random.dirichlet(pi_alpha)
        mu = np.array([np.random.rand(d) * (mu_rng[1] - mu_rng[0]) + mu_rng[0] for _ in range(n)])
        var = np.array([wishart.rvs(df=var_df, scale=var_scale) for _ in range(n)])
        var = var[..., None, None] if d == 1 else var
        out_gm = GM(pi=pi, mu=mu, var=var)

        return out_gm

    def prob(self, t):
        """Return gaussian mixture's probability on t

        Args:
            t: array of (..., D) and D is feature dimension of the mixture

        Returns:
            np.ndarray: same length with t

        """

        if t.shape[-1] != self.d:
            raise ValueError(f'query points have different feature dim ({t.shape[-1]}), not {self.d}')
        prob = np.sum(self.pi * gauss_prob(t, self.mu, self.var), axis=-1)

        return prob

    def merge(self, idx_list: List[Iterable | np.ndarray]):
        """Merge indexed gaussian mixture components with preserved moments

        Args:
            idx_list: list of index for merged components

        """

        flatten_idx = np.hstack(idx_list)
        assert len(flatten_idx) == len(set(flatten_idx)), 'Overlapped indices of components to be merged'

        merge_pi = []
        merge_mu = []
        merge_var = []

        for idx in idx_list:
            idx = list(idx)

            target_pi = np.take(self.pi, idx)
            target_mu = self.mu[idx]
            target_var = self.var[idx]

            _pi = np.sum(target_pi)
            _mu = 1. / _pi * np.sum(target_pi[..., None] * target_mu, axis=0)
            _btw = np.einsum('...i, ...j -> ...ij', target_mu - _mu, target_mu - _mu)
            _var = 1. / _pi * np.sum(target_pi[..., None, None] * (target_var + _btw), axis=0)

            merge_pi.append(_pi)
            merge_mu.append(_mu)
            merge_var.append(_var)

        ori_i = np.setdiff1d(np.arange(self.n), flatten_idx)
        self.n = len(ori_i) + 1
        self.pi = np.hstack([self.pi[ori_i], np.hstack(merge_pi)])
        self.mu = np.vstack([self.mu[ori_i], np.vstack(merge_mu)])
        self.var = np.vstack([self.var[ori_i], np.stack(merge_var, axis=0)])


def calc_ise(gm0: GM, gm1: GM):
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

    x_i = np.expand_dims(gm0.mu, axis=1)
    mu_j = gm1.mu[None, ...]
    var_ij = np.expand_dims(gm0.var, axis=1) + gm1.var

    term0 = - 0.5 * d * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(var_ij))
    term1 = - 0.5 * np.einsum('...i,...i->...', x_i - mu_j, np.linalg.solve(var_ij, x_i - mu_j))
    H = np.exp(term0 + term1)

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

    term0 = np.einsum(
        '...i, ...j -> ...ij',
        np.expand_dims(gm0.mu, axis=1) - gm1.mu,
        np.expand_dims(gm0.mu, axis=1) - gm1.mu,
    )
    term1 = np.expand_dims(gm0.var, axis=1) + term0
    term2 = np.trace(np.linalg.solve(gm1.var, term1) - np.eye(d), axis1=-1, axis2=-2)
    term3 = np.log(np.linalg.det(gm1.var) / np.linalg.det(gm0.var)[..., None])

    kl = 0.5 * (term2 + term3)
    return kl
