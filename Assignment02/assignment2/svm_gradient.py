import numpy as np


def svm_gradient(w, b, x, y, C):
    """
    Compute gradient for SVM w.r.t. to the parameters w and b on a mini-batch (x, y)

    Args:
        w: Parameters of shape [num_features]
        b: Bias (a scalar)
        x: A mini-batch of training example [k, num_features]
        y: Labels corresponding to x of size [k]

    Returns:
        grad_w: The gradient of the SVM objective w.r.t. w of shape [num_features]
        grad_b: The gradient of the SVM objective w.r.t. b of shape [1]

    """

    grad_w = np.zeros(x.shape[1])
    grad_b = 0


    #######################################################################
    # TODO:                                                               #
    # Compute the gradient for a particular choice of w and b.            #
    # Compute the partial derivatives and set grad_w and grad_b to the    #
    # partial derivatives of the cost w.r.t. both parameters              #
    #                                                                     #
    #######################################################################
    f_x = (x).dot(w) + b
    param = 1/C
    distance = 1 - (y*f_x)

    # Gradient w.r.t. w
    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            grad_w += param*w
        else:
            grad_w += param*w - y[ind]*x[ind, :]

    grad_w /= x.shape[0]

    # Gradient w.r.t. b
    for ind, d in enumerate(distance):
        if max(0, d) == 0:
            grad_b += 0
        else:
            grad_b += -y[ind]
    
    grad_b /= x.shape[0]

    #######################################################################
    #                         END OF YOUR CODE                            #
    #######################################################################
    return grad_w, grad_b
