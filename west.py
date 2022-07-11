import numpy as np
from copy import deepcopy

from gm import GM, calc_ise, merge_gm


def fit_west(gm_ori: GM, L: int, gamma=float('inf')):
    """ find a reduced gaussian mixture using West algorithm

    Args:
        gm_ori: the original gaussian mixture
        L: target number of components
        gamma: minimum distance of components to be merged

    Returns:
        GM: merged gaussian mixture

    """

    out_gm = deepcopy(gm_ori)

    while out_gm.n > L:
        min_pi = np.argmin(out_gm.pi / out_gm.var)

        costs = [calc_ise(GM(n=1, pi=np.ones(1), mu=out_gm.mu[min_pi], var=out_gm.var[min_pi]),
                          GM(n=1, pi=np.ones(1), mu=out_gm.mu[j], var=out_gm.var[j]))
                 for j in range(out_gm.n)]
        costs[min_pi] = float('inf')

        min_c_j = np.argmin(costs)

        if costs[min_c_j] < gamma:
            out_gm = merge_gm(out_gm, [[min_pi, min_c_j]])
        else:
            break

    return out_gm
