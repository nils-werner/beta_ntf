import string
import numpy as np


def parafac(factors):
    """Computes the parafac model of a list of matrices

    if factors=[A,B,C,D..Z] with A,B,C..Z of shapes a*k, b*k...z*k, returns
    the a*b*..z ndarray P such that
    p(ia,ib,ic,...iz)=\sum_k A(ia,k)B(ib,k)C(ic,k)...Z(iz,k)

    Parameters
    ----------
    factors : list of arrays
        The factors

    Returns
    -------
    out : array
        The parafac model
    """
    ndims = len(factors)
    request = ''
    for temp_dim in range(ndims):
        request += string.lowercase[temp_dim] + 'z,'
    request = request[:-1] + '->' + string.lowercase[:ndims]
    return np.einsum(request, *factors)


def nnrandn(shape):
    """generates randomly a nonnegative ndarray of given shape

    Parameters
    ----------
    shape : tuple
        The shape

    Returns
    -------
    out : array of given shape
        The non-negative random numbers
    """
    return np.abs(np.random.randn(*shape))
