cimport cython
import string
import time
import numpy as np
cimport numpy as np
from utils import parafac

DTYPE = np.float64
ctypedef np.float64_t DTYPE_t


@cython.binding(True)
def fit(self, np.ndarray X, np.ndarray W=np.array([1])):
    """Learns NTF model

    Parameters
    ----------
    X : ndarray with nonnegative entries
        The input array
    W : ndarray
        Optional ndarray that can be broadcasted with X and
        gives weights to apply on the cost function
    """

    eps = self.eps
    beta = self.beta
    ndims = len(self.data_shape)

    print 'Fitting NTF model with %d iterations....' % self.n_iter

    # main loop
    for it in range(self.n_iter):
        if self.verbose:
            if 'tick' not in locals():
                tick = time.time()
            print ('cython NTF model, iteration %d / %d, duration=%.1fms, cost=%f'
                   % (it, self.n_iter, (time.time() - tick) * 1000,
                      self.score(X)))
            tick = time.time()

        #updating each factor in turn
        for dim in range(ndims):
            if dim in self.fixed_factors:
                continue

            # get current model
            model = parafac(self.factors_)

            # building request for this update to use with einsum
            # for exemple for 3-way tensors, and for updating factor 2,
            # will be : 'az,cz,abc->bz'
            request = ''
            operand_factors = []
            for temp_dim in range(ndims):
                if temp_dim == dim:
                    continue
                request += string.lowercase[temp_dim] + 'z,'
                operand_factors.append(self.factors_[temp_dim])
            request += string.lowercase[:ndims] + '->'
            request += string.lowercase[dim] + 'z'
            # building data-dependent factors for the update
            operand_data_numerator = [X * W * (model[...] ** (beta - 2.))]
            operand_data_denominator = [W * (model[...] ** (beta - 1.))]
            # compute numerator and denominator for the update
            numerator = eps + np.einsum(request, *(
                operand_factors + operand_data_numerator))
            denominator = eps + np.einsum(request, *(
                operand_factors + operand_data_denominator))
            # multiplicative update
            self.factors_[dim] *= numerator / denominator
    print 'Done.'
    return self
