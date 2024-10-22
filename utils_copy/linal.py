"""
Я испугался собственных значений
"""
import numpy as np
from numpy import linalg as LA



def SVD(A: np.array, full_matrix: bool = True) -> (np.array, np.array, np.array):
    """
    implemented Principal Component Analysis (eig used from numpy)
    parameters 
    X - np.array, shape (n, m) 

    returns svd decomposition
    U - np.array shape (n, n)
    D - np.array shape (n, m)
    V - np.array shape (m, m)
    """
    n = A.shape[0]
    m = A.shape[1]
    if n >= m:
        matrix1 = A.T @ A #shape: (m, m)
        matrix2 = A @ A.T #shape: (n, n)
        eigenvalues1, eigenvectors1 = LA.eig(matrix1)
        eigenvalues2, eigenvectors2 = LA.eig(matrix2)


        index1 = np.argsort(eigenvalues1, axis=0)[::-1]
        index2 = np.argsort(eigenvalues2, axis=0)[::-1]
        V = eigenvectors1[:, index1]
        U = eigenvectors2[:, index2]
        U = U.real
        V = V.real
        D = np.sqrt(eigenvalues1[index1])
        # D = np.diag(np.sqrt(eigenvalues1[index1]))
        # D = np.concatenate((D, np.zeros([n-m, m])), axis=0)
        if full_matrix:
            return U, D, V.T
        else:
            return U[:, :m], D, V.T
    else:
        return SVD(A.T, full_matrix=full_matrix)
    

def svd_flip(u, v, u_based_decision=True):
    """Sign correction to ensure deterministic output from SVD.

    Adjusts the columns of u and the rows of v such that the loadings in the
    columns in u that are largest in absolute value are always positive.

    Parameters
    ----------
    u : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.

    v : ndarray
        Parameters u and v are the output of `linalg.svd` or
        :func:`~sklearn.utils.extmath.randomized_svd`, with matching inner
        dimensions so one can compute `np.dot(u * s, v)`.
        The input v should really be called vt to be consistent with scipy's
        output.

    u_based_decision : bool, default=True
        If True, use the columns of u as the basis for sign flipping.
        Otherwise, use the rows of v. The choice of which variable to base the
        decision on is generally algorithm dependent.

    Returns
    -------
    u_adjusted : ndarray
        Array u with adjusted columns and the same dimensions as u.

    v_adjusted : ndarray
        Array v with adjusted rows and the same dimensions as v.
    """
    if u_based_decision:
        # columns of u, rows of v
        max_abs_cols = np.argmax(np.abs(u), axis=0)
        signs = np.sign(u[max_abs_cols, range(u.shape[1])])
        u *= signs
        v *= signs[:, np.newaxis]
    else:
        # rows of v, columns of u
        max_abs_rows = np.argmax(np.abs(v), axis=1)
        signs = np.sign(v[range(v.shape[0]), max_abs_rows])
        u *= signs
        v *= signs[:, np.newaxis]
    return u, v

