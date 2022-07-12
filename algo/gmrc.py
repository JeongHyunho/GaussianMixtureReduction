from copy import deepcopy

import numpy as np

from mixtures.gm import GM, kl_gm_comp, merge_gm
from algo.runnalls import fit_runnalls


def fit_gmrc(gm_ori: GM, L: int):
    """Find a reduced mixture via clustering after initialization by Runnalls' alg

    Args:
        gm_ori: the original gaussian mixture
        L: the number of components of reduced mixture

    Returns:
        GM: reduced gaussian mixture

    """

    out_gm = fit_runnalls(gm_ori, L=L)
    out_gm = kmeans_gm(gm_ori, out_gm)

    return out_gm


def kmeans_gm(gm0: GM, gm1: GM):
    """Merge gm0 to have the same number of components with gm1 by kmeans
    Two components of gm0 closest to gm1' component is merged iteratively

    Args:
        gm0: mixture which has N components and shrinks
        gm1: mixture used to form clusters

    Returns:
        GM: merged gaussian mixture

    """

    old_gm = deepcopy(gm0)
    out_gm = deepcopy(gm1)

    while old_gm != out_gm:
        old_gm = out_gm

        kl = kl_gm_comp(gm0, gm1)
        alloc_idx = np.argmin(kl, axis=-1)
        cluster_idx, num_el = np.unique(alloc_idx, return_counts=True)

        merge_idx = []
        for idx in cluster_idx[num_el > 1]:
            merge_idx.append((alloc_idx == idx).nonzero()[0])

        out_gm = merge_gm(gm0, merge_idx)

    return out_gm
