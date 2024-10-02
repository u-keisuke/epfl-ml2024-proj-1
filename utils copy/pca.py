import numpy as np
from .linal import SVD, svd_flip
import numpy.linalg as LA
import matplotlib.pyplot as plt
import scipy.linalg as scipyLA

class PCA():
    """
    implemented Principal Component Analysis
    parameters 
    X - np.array, shape (n, m) 
    k - int, number of components

    methods:
    reduction - to return the reduced data
    plot_sing_vals - plotting all singular values of X.T @ X
    plot_eiguenvalues - only 2-dimensional case (drawing the data + elips for eig vals and vectors)
    """
    def __init__(self, X) -> None:
        self.X = X

    def normalize(self, choice: {"full", "std", "mean"}):
        if choice == "full":
            self.X = (self.X - np.mean(self.X, axis=0)) / np.std(self.X, axis=0)
        elif choice == "std":
            self.X = self.X / np.std(self.X, axis=0)
        elif choice == "mean":
            self.X = (self.X - np.mean(self.X, axis=0)) 
        else:
            raise ValueError("Unknown argument `%s`" % choice)

    
    def reduction(self, k):
        """
        up to some rescaling ok
        """
        n, m = self.X.shape
        k = min(m, n, k)
        #projection on eigvectors from V
        # _, _, V = LA.svd(self.X)
        # return self.X @ V[:, :k]

        # U, val, Vt = scipyLA.svd(self.X, full_matrices=False)
        # U, Vt = svd_flip(U, Vt)

        U, val, Vt = SVD(self.X, full_matrix=False)
        U, Vt = svd_flip(U, Vt)

        return U[:, :k] * val[:k]
     
    def plot_sing_vals(self, scale=None):
        _, vals, _ = np.linalg.svd(self.X, full_matrices=False)
        plt.figure(figsize=(12,7))
        plt.plot(vals,'ro')
        if scale is not None:
            plt.yscale(scale)
        plt.ylabel("Singular values")
        plt.xlabel("Singular value order")
        plt.show()

    def plot_eiguenvalues(self):
        assert self.X.shape[1] ==2
        U, eigenvalues, Vt = LA.svd(self.X / 15) #(self.X.shape[0]-1)
        eigenvalues = eigenvalues 
        # matrix1 = self.X.T @ self.X
        eigenvectors = Vt.T


        vector1 = eigenvectors[:, 0] * eigenvalues[0] #np.sqrt()
        vector2 = eigenvectors[:, 1] * eigenvalues[1] #  np.sqrt()
        theta = np.linspace(0, 2 * np.pi, 1000)
        ellipsis = (eigenvectors * eigenvalues[None,:]) @ [np.sin(theta), np.cos(theta)] #np.sqrt()
        # print(eigenvectors * np.sqrt(eigenvalues[None,:]) )
        plt.figure(figsize=(12,7))
        plt.plot(ellipsis[0,:], ellipsis[1,:], color='yellow')
        plt.scatter(self.X.T[0], self.X.T[1])
        plt.plot(np.array([0, vector1[0]]), np.array([0, vector1[1]]), color="g")
        plt.plot(np.array([0, vector2[0]]), np.array([0, vector2[1]]), color="r")
        plt.show()

# x = np.linspace(1, 4, 1000)
# y =  np.linspace(1, 4, 1000) + np.random.normal(0, 1, size=1000) 
# # y = -np.linspace(1, 10, 100)

# A = np.stack((x, y), axis=0)
# pca = PCA(A.T)
# pca.normalize("full")
# pca.plot_eiguenvalues()  
# vectors = np.array([[1, 0.5],
#                     [0,1]])
# vector1 = vectors[:, 0]
# print(vector1)
# vector2 = vectors[:, 1]
# print(vector2)
# theta = np.linspace(0, 2 * np.pi, 1000)
# ellipsis = (vectors ) @ [np.sin(theta), np.cos(theta)] #np.sqrt()
# plt.figure(figsize=(12,7))
# plt.plot(ellipsis[0,:], ellipsis[1,:], color='yellow')
# plt.plot(np.array([0, vector1[0]]), np.array([0, vector1[1]]), color="g")
# plt.plot(np.array([0, vector2[0]]), np.array([0, vector2[1]]), color="r")
# plt.show()




