import numpy as np
from beta_ntf import *
from beta_ntf.utils import nnrandn

if __name__ == '__main__':
    # Choosing the shape of the data to approximate (tuple of length up to 25)
    # data_shape = (1000, 400, 10)  # 3-way tensor
    data_shape = (1000, 400)  # matrix
    data_shape = (50, 5, 10, 6, 7)  # 5-way tensor
    #data_shape = (1000, 400)  # matrix

    # Choosing the number of components for testing
    n_components = 9

    # Building the true factors to generate data
    factors = [nnrandn((shape, n_components)) for shape in data_shape]

    # Generating the data through the parafac function
    V = parafac(factors)

    # Create BetaNTF object
    beta_ntf = BetaNTF(V.shape, n_components=10, beta=1, n_iter=100,
                       verbose=True)

    # Fit the model
    beta_ntf.fit(V)

    #Print resulting score
    print 'Resulting score', beta_ntf.score(V)
    print 'Compression ratio : %0.1f%%'%((1.-sum(beta_ntf.data_shape)*
                        beta_ntf.n_components
                        /float(prod(beta_ntf.data_shape)))*100.)
        

    #Now illustrate the get model
    total_model= parafac(beta_ntf.factors_)
    two_components = beta_ntf[...,:2]
    print 'Shape of total_model : ',total_model.shape
    print 'Shape of two_components : ',two_components.shape
