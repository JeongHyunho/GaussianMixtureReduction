import numpy as np


def gauss_prob(x: np.ndarray, mu: np.ndarray, var: np.ndarray) -> np.ndarray:
    """Return probability of x for (mu, var) distributed Gaussian

    Args:
        x: array of (..., D)
        mu: array of (N, D)
        var: array of (N, D, D)

    Returns:
        np.ndarray: array of (..., N)

    """

    d = x.shape[-1]
    ex_x = np.expand_dims(x, axis=-2)
    ex_var = np.broadcast_to(var, x.shape[:-1] + var.shape)

    term0 = - 0.5 * d * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(var))
    term1 = - 0.5 * np.einsum('...i,...i->...', ex_x - mu, np.linalg.solve(ex_var, ex_x - mu))
    prob = np.exp(term0 + term1)

    return prob


def integral_prod_guass_prob(mu0, var0, mu1, var1, mode='self'):
    """Return integration of product of two gaussian

    Args:
        mu0: array of (N, D)
        var0: array of (N, D, D)
        mu1: array of (M, D)
        var1: array of (M, D, D)
        mode: 'self' or 'cross'

    Returns:
        np.ndarray: array of (N,) if self, (N, M) otherwise

    """

    d = mu0.shape[-1]

    if mode == 'self':
        diff_mu_ij = mu0 - mu1                                                              # (..., N, D)
        sum_var_ij = var0 + var1                                                            # (..., N, D, D)
    elif mode == 'cross':
        diff_mu_ij = np.expand_dims(mu0, axis=-2) - np.expand_dims(mu1, axis=-3)            # (..., N, M, D)
        sum_var_ij = np.expand_dims(var0, axis=-3) + np.expand_dims(var1, axis=-4)          # (..., N, M, D, D)
    else:
        raise ValueError(f"mode(:{mode}) should be in ['self', 'cross'].")

    term0 = - 0.5 * d * np.log(2 * np.pi) - 0.5 * np.log(np.linalg.det(sum_var_ij))
    term1 = - 0.5 * np.einsum('...i,...i->...', diff_mu_ij, np.linalg.solve(sum_var_ij, diff_mu_ij))
    prob = np.exp(term0 + term1)

    return prob
