import numpy as np
import time
from .svm_loss import svm_loss
from .svm_gradient import svm_gradient


def svm_solver(X, y, C, num_iter=5000, num_per_batch=32):
    """
    Pegasos SVM solver.

    Args:
        X: Data matrix of shape [num_train, num_features]
        y: Labels corresponding to X of size [num_train]
        C: SVM hyper-parameter
        num_iter: Number of iterations
        num_per_batch: Number of samples per mini-batch

    Returns:
        w: The learnt weights vector
        b: The learnt intercept

    """

    w = np.zeros(X.shape[1])
    b = 0.
    param = 1/C
    f_x = X.dot(w) + b
    for t in range(1,num_iter):
        start = time.time()
        #######################################################################
        # TODO:                                                               #
        # Perform one step of stochastic gradient descent:                    #
        #   - Select a single training example at random                      #
        #   - Update theta based on alpha and using gradient_function         #
        #                                                                     #
        #######################################################################

        # 1st Step: Sample a random mini-batch of size num_per_batch
        mini_batch = np.random.choice(X.shape[0], num_per_batch, replace=False)
        x_batch = X[mini_batch, :]
        y_batch = y[mini_batch]

        # 2nd Step: Compute the learning-rate n_t=1/(lambda*t) where lambda=1/C
        n_t = 1/(param*t)


        # 3rd Step: Compute the gradients and update the parameters as
        # w:=w-n_t*grad_w and b:=b-n_t*grad_b
        grad_w, grad_b = svm_gradient(w, b, x_batch, y_batch, C)
        w -= n_t * grad_w
        b -= n_t * grad_b

        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################
        if t % 5000 == 0:
            exec_time = time.time() - start
            loss = svm_loss(w, b, X, y, C)
            print('Iter {}/{}: cost = {}  ({}s)'.format(t, num_iter, loss, exec_time))

    return w, b
