# You should take care of the following:

# Return type: 
# - Note that all functions should return: (w, loss), which is the last weight vector of the method, and the corresponding loss value (cost function). 
# - Note that while in previous labs you might have kept track of all encountered w for iterative methods, here we only want the last one. 
# - Moreover, the loss returned by the regularized methods (ridge_regression and reg_logistic_regression) should not include the penalty term.
# File names: 
# - Please provide all function implementations in a single python file, called implementations.py.
# All code should be easily readable and commented.
# Note that we will call your provided methods and evaluate for correct implementation. 
# We provide some basic tests to check your implementation in https://github.com/epfml/ML_course/tree/main/projects/project1/grading_tests.


import numpy as np

def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute loss by MSE
    # ***************************************************
    return 1/2/len(y)*sum((y - tx @ w) ** 2)
def compute_stoch_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """

    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: implement stochastic gradient computation. It's the same as the usual gradient.
    # ***************************************************
    return -1./len(y) * tx.T @ (y - tx @ w)

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    # ***************************************************
    # INSERT YOUR CODE HERE
    # TODO: compute gradient vector
    # ***************************************************
    return -1./len(y) * tx.T @ (y - tx @ w)

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.
    """
    # Define parameters to store w and loss
    ws = [initial_w]
    losses = []
    w = initial_w
    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: compute gradient and loss
        # ***************************************************
        grad, loss = compute_gradient(y, tx, w), compute_loss(y,tx,w)
        
        # ***************************************************
        #raise NotImplementedError
        # ***************************************************
        # INSERT YOUR CODE HERE
        w = w.astype(np.float64)
        w -= gamma * grad
        # store w and loss
    return w, loss



def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.
    """
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # ***************************************************
        # INSERT YOUR CODE HERE
        # TODO: implement stochastic gradient descent.
        # ***************************************************
        loss = compute_loss(y,tx,w)
        for y_batch,tx_batch in batch_iter(y, tx, batch_size):
            w = w.astype(np.float64)
            w -= gamma * compute_stoch_gradient(y_batch,tx_batch,w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations.
    """
    w =  (np.linalg.inv(tx.T @ tx)) @ tx.T @ y
    #w = np.linalg.pinv(tx) @ y
    return w, compute_loss(y, tx,w)

def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.
    """
    w = np.linalg.inv(tx.T @ tx + lambda_ * (2 * len(y)) * np.eye(tx.shape[1])) @ tx.T @ y
    return w, compute_loss(y,tx,w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent. (y in {-1, 1})
    """
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent. (y in {-1, 1} with regularization term lambda_*||w||^2)
    """
    raise NotImplementedError