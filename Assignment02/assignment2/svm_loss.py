import numpy as np


def svm_loss(w, b, X, y, C):
    """
    Computes the loss of a linear SVM w.r.t. the given data and parameters

    Args:
        w: Parameters of shape [num_features]
        b: Bias (a scalar)
        X: Data matrix of shape [num_data, num_features]
        y: Labels corresponding to X of size [num_data, 1]
        C: SVM hyper-parameter

    Returns:
        l: The value of the objective for a linear SVM

    """

    
    #######################################################################
    # TODO:                                                               #
    # Compute and return the value of the unconstrained SVM objective     #
    #                                                                     #
    #######################################################################
    f_x = X.dot(w) + b
    param = 1/C
    regul_term = param/2 * np.sum(w*w)
    distance = 1 - (y*f_x)
    l = np.mean(np.maximum(0, distance))
    l += regul_term

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return l
