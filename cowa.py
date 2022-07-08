import numpy as np

from cvxopt import solvers, matrix

from gm import GM, calc_integral_outer_prod_gm
from west import fit_west


solvers.options['show_progress'] = False


def fit_cowa(gm_ori: GM, L: int):
    """Find a reduced mixture via constraint optimized weight adaptation

    Args:
        gm_ori: the original gaussian mixture
        L: the number of components of reduced mixture

    Returns:
        GM: reduced gaussian mixture

    """

    fit_gm = fit_west(gm_ori, L=L)

    H0 = calc_integral_outer_prod_gm(fit_gm, fit_gm)
    H1 = calc_integral_outer_prod_gm(gm_ori, fit_gm)

    P = matrix(H0)
    q = matrix(- H1.T @ gm_ori.pi)
    G = matrix(- np.eye(fit_gm.n))
    h = matrix(0.0, (fit_gm.n, 1))
    A = matrix(1.0, (1, fit_gm.n))
    b = matrix(1.0)

    sol = solvers.qp(P, q, G, h, A, b)
    out_gm = GM(n=fit_gm.n, pi=np.array(sol['x']).flatten(), mu=fit_gm.mu, var=fit_gm.var)

    return out_gm
