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


def batch_iter(y, tx, batch_size, num_batches=1, shuffle=True):
    """
    Generate a minibatch iterator for a dataset.
    Takes as input two iterables (here the output desired values 'y' and the input data 'tx')
    Outputs an iterator which gives mini-batches of `batch_size` matching elements from `y` and `tx`.
    Data can be randomly shuffled to avoid ordering in the original data messing with the randomness of the minibatches.

    Example:

     Number of batches = 9

     Batch size = 7                              Remainder = 3
     v     v                                         v v
    |-------|-------|-------|-------|-------|-------|---|
        0       7       14      21      28      35   max batches = 6

    If shuffle is False, the returned batches are the ones started from the indexes:
    0, 7, 14, 21, 28, 35, 0, 7, 14

    If shuffle is True, the returned batches start in:
    7, 28, 14, 35, 14, 0, 21, 28, 7

    To prevent the remainder datapoints from ever being taken into account, each of the shuffled indexes is added a random amount
    8, 28, 16, 38, 14, 0, 22, 28, 9

    This way batches might overlap, but the returned batches are slightly more representative.

    Disclaimer: To keep this function simple, individual datapoints are not shuffled. For a more random result consider using a batch_size of 1.

    Example of use :
    for minibatch_y, minibatch_tx in batch_iter(y, tx, 32):
        <DO-SOMETHING>
    """
    data_size = len(y)  # NUmber of data points.
    batch_size = min(data_size, batch_size)  # Limit the possible size of the batch.
    max_batches = int(
        data_size / batch_size
    )  # The maximum amount of non-overlapping batches that can be extracted from the data.
    remainder = (
        data_size - max_batches * batch_size
    )  # Points that would be excluded if no overlap is allowed.

    if shuffle:
        # Generate an array of indexes indicating the start of each batch
        idxs = np.random.randint(max_batches, size=num_batches) * batch_size
        if remainder != 0:
            # Add an random offset to the start of each batch to eventually consider the remainder points
            idxs += np.random.randint(remainder + 1, size=num_batches)
    else:
        # If no shuffle is done, the array of indexes is circular.
        idxs = np.array([i % max_batches for i in range(num_batches)]) * batch_size

    for start in idxs:
        start_index = start  # The first data point of the batch
        end_index = (
            start_index + batch_size
        )  # The first data point of the following batch
        yield y[start_index:end_index], tx[start_index:end_index]


def compute_loss(y, tx, w):
    """Calculate the loss using either MSE or MAE.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2,). The vector of model parameters.

    Returns:
        the value of the loss (a scalar), corresponding to the input parameters w.
    """
    # one liner to compute los
    return 1 / 2 / len(y) * sum((y - tx @ w) ** 2)


def compute_gradient(y, tx, w):
    """Compute a stochastic gradient at w from just few examples n and their corresponding y_n labels.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the stochastic gradient of the loss at w.
    """
    # one liner to compute gradient
    return -1.0 / len(y) * tx.T @ (y - tx @ w)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using gradient descent."""
    # Define parameters to store w and loss
    ws = [initial_w]
    w = initial_w
    for n_iter in range(max_iters):
        # computing gradient and loss
        grad = compute_gradient(y, tx, w)
        # grad step
        w = w.astype(np.float64)
        w -= gamma * grad
        # store w and loss
    return w, compute_loss(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using stochastic gradient descent."""
    ws = [initial_w]
    losses = []
    w = initial_w

    for n_iter in range(max_iters):
        # computing loss
        for y_batch, tx_batch in batch_iter(y, tx, 1):
            w = w.astype(np.float64)
            w -= gamma * compute_gradient(y_batch, tx_batch, w)
    loss = compute_loss(y, tx, w)
    return w, loss


def least_squares(y, tx):
    """Least squares regression using normal equations."""
    # least squqres formula
    w = (np.linalg.inv(tx.T @ tx)) @ tx.T @ y
    return w, compute_loss(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression
    using normal equations.
    """
    # one liner for ridge regression
    w = (
        np.linalg.inv(tx.T @ tx + lambda_ * (2 * len(y)) * np.eye(tx.shape[1]))
        @ tx.T
        @ y
    )
    return w, compute_loss(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression using gradient descent. (y in {-1, 1})"""

    def sigmoid(t):
        return 1 / (1 + np.exp(-t))

    w = initial_w
    for _ in range(max_iters):
        pred = tx @ w
        gradient = -tx.T @ (y * sigmoid(-y * pred))
        w = w - gamma * gradient
    # Compute the final loss
    loss = np.sum(np.log(1 + np.exp(-y * (tx @ w))))

    return w, loss


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized logistic regression using gradient descent. (y in {-1, 1} with regularization term lambda_*||w||^2)"""

    def sigmoid(t):
        return 1 / (1 + np.exp(-t))

    w = initial_w
    for _ in range(max_iters):
        pred = tx @ w
        gradient = -tx.T @ (y * sigmoid(-y * pred)) + 2 * lambda_ * w
        w = w - gamma * gradient
    # Compute the final loss
    loss = np.sum(np.log(1 + np.exp(-y * (tx @ w)))) + lambda_ * np.linalg.norm(w) ** 2

    return w, loss
