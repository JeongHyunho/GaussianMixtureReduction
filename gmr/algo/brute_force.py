from copy import deepcopy

import numpy as np

from gmr.mixtures.gm import GM, calc_ise


def fit_brute_force(gm_ori: GM, L: int):
    """Find optimal merging partitions to reduce gaussian mixtures by brute-force

    Args:
        gm_ori: target gaussian mixture to be reduced
        L: the number of components of fitted mixture

    Returns:
        GM: reduced gaussian mixture

    """

    best_ise = float('inf')
    out_gm = None

    for parts in set_partitions(np.arange(gm_ori.n), L):
        non_singleton = [p for p in parts if len(p) > 1]
        m_gm = deepcopy(gm_ori)
        m_gm.merge(non_singleton)
        ise = calc_ise(gm_ori, m_gm)

        if ise < best_ise:
            best_ise = ise
            out_gm = m_gm

    return out_gm


def set_partitions(iterable, k=None):
    """
    Yield the set partitions of *iterable* into *k* parts. Set partitions are
    not order-preserving.

    credit for https://github.com/more-itertools/more-itertools
    """

    L = list(iterable)
    n = len(L)
    if k is not None:
        if k < 1:
            raise ValueError(
                "Can't partition in a negative or zero number of groups"
            )
        elif k > n:
            return

    def set_partitions_helper(L, k):
        n = len(L)
        if k == 1:
            yield [L]
        elif n == k:
            yield [[s] for s in L]
        else:
            e, *M = L
            for p in set_partitions_helper(M, k - 1):
                yield [[e], *p]
            for p in set_partitions_helper(M, k):
                for i in range(len(p)):
                    yield p[:i] + [[e] + p[i]] + p[i + 1 :]

    if k is None:
        for k in range(1, n + 1):
            yield from set_partitions_helper(L, k)
    else:
        yield from set_partitions_helper(L, k)
