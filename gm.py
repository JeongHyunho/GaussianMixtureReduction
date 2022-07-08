from dataclasses import dataclass
from typing import List, Iterable

import numpy as np
from scipy.stats import wishart


@dataclass
class GM:
    """ 1D Gaussian Mixture """

    n: int
    pi: np.ndarray
    mu: np.ndarray
    var: np.ndarray

    def __post_init__(self):
        # one component gm handling
        if self.n == 1:
            self.pi = np.array([self.pi]) if np.isscalar(self.pi) else self.pi
            self.mu = np.array([self.mu]) if np.isscalar(self.mu) else self.mu
            self.var = np.array([self.var]) if np.isscalar(self.var) else self.var

        # check dim
        assert self.n == len(self.pi)
        assert self.n == len(self.mu)
        assert self.n == len(self.var)

    def __eq__(self, other):
        if self.n != other.n:
            return False
        elif np.any(self.mu != other.mu):
            return False
        elif np.any(self.var != other.var):
            return False
        else:
            return True


def sample_gm(n, pi_alpha, mu_rng, var_df, var_scale, seed=None):
    """Sample specified gaussian mixture
    mean from uniform, var from wishert distribution

     Returns:
         GM: sampled mixture

     """

    assert len(mu_rng) == 2 and mu_rng[0] <= mu_rng[1], f'mu_rng of [min, max] is expected, but {mu_rng}'

    if seed is not None:
        np.random.seed(seed)

    pi = np.random.dirichlet(pi_alpha)
    mu = np.array([np.random.rand() * (mu_rng[1] - mu_rng[0]) + mu_rng[0] for _ in range(n)])
    var = np.array([wishart.rvs(df=var_df, scale=var_scale) for _ in range(n)])
    out_gm = GM(n=n, pi=pi, mu=mu, var=var)

    return out_gm
    

def gm_prob(t, gm: GM):
    """Return gaussian mixture's probability on t

    Returns:
        np.ndarray: same length with t

    """

    log_prob_i = - 0.5 * (np.log(2 * np.pi) + np.log(gm.var) + (t[..., None] - gm.mu) ** 2 / gm.var)
    prob = np.sum(gm.pi * np.exp(log_prob_i), axis=-1)

    return prob


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
    """Integrates the outer prodeuct of two gaussian mixtures

    Returns:
        np.ndarray: n by n matrix

    """

    x_ij = gm0.mu[..., None]
    mu_ij = gm1.mu[None, ...]
    var_ij = gm0.var[..., None] + gm1.var[None, ...]

    log_H = - 0.5 * (np.log(2 * np.pi) + np.log(var_ij) + (x_ij - mu_ij) ** 2 / var_ij)
    H = np.exp(log_H)

    return H


def merge_gm(gm: GM, idx_list: List[Iterable | np.ndarray]):
    """Merge indexed gaussian mixtures component with preserved moments

    Args:
        gm: gaussian mixture to be merged
        idx_list: list of index for merged components

    Returns:
        GM: merged gaussian mixture

    """

    flatten_idx = np.hstack(idx_list)
    assert len(flatten_idx) == len(set(flatten_idx)), 'Overlapped indices of components to be merged'

    n = gm.n
    merge_pi = []
    merge_mu = []
    merge_var = []

    for idx in idx_list:
        target_pi = np.take(gm.pi, idx)
        target_mu = np.take(gm.mu, idx)
        target_var = np.take(gm.var, idx)

        n = n - len(idx) + 1
        _pi = np.sum(target_pi)
        _mu = 1. / _pi * np.sum(target_pi * target_mu)
        _var = 1. / _pi * np.sum(target_pi * (target_var + (target_mu - _mu) ** 2))

        merge_pi.append(_pi)
        merge_mu.append(_mu)
        merge_var.append(_var)

    ori_i = np.setdiff1d(np.arange(gm.n), flatten_idx)
    pi = np.hstack([np.take(gm.pi, ori_i), np.hstack(merge_pi)])
    mu = np.hstack([np.take(gm.mu, ori_i), np.hstack(merge_mu)])
    var = np.hstack([np.take(gm.var, ori_i), np.hstack(merge_var)])

    out_gm = GM(n=n, pi=pi, mu=mu, var=var)
    return out_gm


def kl_gm_comp(gm0: GM, gm1: GM):
    """Calculates KL divergence between all components in two gaussian mixtures

    Args:
        gm0: mixture which has N components
        gm1: mixture which has M components

    Returns:
        np.ndarray: N by M matrix, (i, j) element is kl btw i-th and j-th components from gm0 and gm1 respectively

    """

    kl = np.log(gm1.var / gm0.var[..., None]) \
         + (gm0.var[..., None] - gm1.var + (gm0.mu[..., None] - gm1.mu) ** 2) / gm1.var

    return kl
