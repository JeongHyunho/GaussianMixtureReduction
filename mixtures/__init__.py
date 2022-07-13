import numpy as np


def check_var(var: np.ndarray):
    """Confirm variance matrix is symmetric and positive definite

    Args:
        var: array of (..., D, D)

    Raises:
        ValueError: if inadequate variance matrix

    """

    _trs_diff = np.abs(var - np.swapaxes(var, -2, -1))
    assert np.all(_trs_diff < 1e-9), 'not symmetric'
    _eigvals = np.linalg.eigvalsh(var)
    assert np.all(_eigvals >= -1e-9), 'not semi-positive definite'


def check_dim(pi: np.ndarray, mu: np.ndarray, var: np.ndarray, batch_form=False):
    """Confirm mixture parameters have consistent dimensions

    Args:
        pi: array of (..., N)
        mu: array of (..., N, D)
        var: array of (..., N, D, D)

    Returns:
        int: the number of components
        int: feature dimension

    Raises:
        ValueError: if mixture parameters' dimensions aren't consistent

    """

    assert pi.ndim + 2 == mu.ndim + 1 == var.ndim, 'wrong batch setup'
    assert pi.shape[-1] == mu.shape[-2] == var.shape[-3], 'wrong number of components'
    assert mu.shape[-1] == var.shape[-2] == var.shape[-1], 'wrong feature dim'

    n = pi.shape[-1]
    d = mu.shape[-1]

    return n, d


def check_batch(pi: np.ndarray, mu: np.ndarray, var: np.ndarray, batch_form=False):
    """Confirm parameters' batch size is consistent

    Args:
        pi: array of (B, N)
        mu: array of (B, N, D)
        var: array of (B, N, D, D)
        batch_form: bool

    Returns:
        int: batch size

    Raises:
        ValueError: if mixture parameters' dimensions aren't consistent

    """

    if batch_form:
        assert pi.ndim > 1 or mu.ndim > 2 or var.ndim > 3, 'batch parameters is not expected'
        assert pi.shape[0] == mu.shape[0] == var.shape[0], 'inconsistent batch size'

        return pi.shape[0]

    else:
        assert pi.ndim < 2 and mu.ndim < 3 and var.ndim < 4, 'only 1D batch is allowed'

        return None
