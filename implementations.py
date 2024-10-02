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


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent.
    """
    raise NotImplementedError


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent.
    """
    raise NotImplementedError


def least_squares(y, tx):
    """Least squares regression using normal equations.
    """
    raise NotImplementedError


def ridge_regression(y, tx, lambda_):
    """Ridge regression using normal equations.
    """
    raise NotImplementedError


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent. (y in {-1, 1})
    """
    raise NotImplementedError


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent. (y in {-1, 1} with regularization term lambda_*||w||^2)
    """
    raise NotImplementedError