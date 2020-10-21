from .sigmoid import sigmoid
import numpy as np


def cost_function(theta, X, y):
    """
    Computes the cost of using theta as the parameter for logistic regression

    Args:
        theta: Parameters of shape [num_features]
        X: Data matrix of shape [num_data, num_features]
        y: Labels corresponding to X of size [num_data, 1]

    Returns:
        l: The cost for logistic regression

    """

    
    #######################################################################
    # TODO:                                                               #
    # Compute and return the log-likelihood l of a particular choice of   #
    # theta.                                                              #
    #                                                                     #
    #######################################################################
    
    # Weighted sum of inputs
    w_sum = np.dot(X, theta)
    h = sigmoid(w_sum)
    l =  np.sum(y * np.log(h) + (1 - y) * np.log(1 - h))

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return l
