from copy import deepcopy

import numpy as np

from cvxopt import solvers, matrix

from mixtures.gm import GM, calc_integral_outer_prod_gm
from algo.west import fit_west


solvers.options['show_progress'] = False


def fit_cowa(gm_ori: GM, L: int, gamma=float('inf')):
    """Find a reduced mixture via constraint optimized weight adaptation

    Args:
        gm_ori: the original gaussian mixture
        L: the number of components of reduced mixture
        gamma: distance threshold

    Returns:
        GM: reduced gaussian mixture

    """

    out_gm = deepcopy(gm_ori)

    while out_gm.n > L:
        old_n = out_gm.n
        out_gm = fit_west(out_gm, L=out_gm.n-1, gamma=gamma)

        # distance threshold
        if out_gm.n == old_n:
            break

        H0 = calc_integral_outer_prod_gm(out_gm, out_gm)
        H1 = calc_integral_outer_prod_gm(gm_ori, out_gm)

        P = matrix(H0)
        q = matrix(- H1.T @ gm_ori.pi)
        G = matrix(- np.eye(out_gm.n))
        h = matrix(0.0, (out_gm.n, 1))
        A = matrix(1.0, (1, out_gm.n))
        b = matrix(1.0)

        sol = solvers.qp(P, q, G=G, h=h, A=A, b=b)
        out_gm = GM(n=out_gm.n, d=out_gm.d, pi=np.array(sol['x']).flatten(), mu=out_gm.mu, var=out_gm.var)

    return out_gm
