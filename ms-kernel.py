import numpy as np
from scipy.sparse.linalg import eigs
import scipy.sparse as sp

def laplacian(W):
    '''
    Normalized graph Laplacian function.
    :param W: np.ndarray, [n_route, n_route], weighted adjacency matrix of G, original adjacency matrix.
    :return: np.matrix, [n_route, n_route].
    '''
    # d ->  diagonal degree matrix
    n, d = np.shape(W)[0], np.sum(W, axis=1)
    # L -> graph Laplacian
    L = -W
    L[np.diag_indices_from(L)] = d
    for i in range(n):
        for j in range(n):
            if (d[i] > 0) and (d[j] > 0):
                L[i, j] = L[i, j] / np.sqrt(d[i] * d[j])
    # lambda_max \approx 2.0, the largest eigenvalues of L.
    lambda_max = eigs(L, k=1, which='LR')[0][0].real
    return np.mat(2 * L / lambda_max - np.identity(n))

def ms_kernel(L, K, n):
    '''
    Chebyshev polynomials approximation function.
    :param L: np.matrix, [n_route, n_route], graph Laplacian.
    :param K: int, kernel size of spatial convolution.
    :param n: int, number of routes / size of graph.
    :return: np.ndarray, [n_route, Ks*n_route].
    '''
    L0, L1 = np.mat(np.identity(n)), np.mat(np.copy(L))
    L1 = L1 - np.diag(np.diag(L1))
    #L1_,L0_ :temp varialble matrix
    L1_ = L1 != 0
    L0_ = L0 != 0

    if K > 1:
        #L_list: save each  order k-multi-step adjacency matrix
        L_list = [np.copy(L0), np.copy(L1)]
        for i in range(K - 2):
            Ln = np.mat(2 * L * L1 - L0)
            #Lnc: set the nonzero elements of Ln to 1
            Ln_ = Ln != 0
            Lnc = np.zeros_like(Ln)
            Lnc[Ln_] = 1
            #L1c: set the nonzero elements of L1 to 1
            L1c = np.zeros_like(L1)
            L1c[L1_] = 1
            #L0c: set the nonzero elements of L0 to 1
            L0c = np.zeros_like(L0)
            L0c[L0_] = 1
            #Lnn: Ln remove two step and one step adjacency matrix
            Lnc = Lnc - L0c
            Lnc[Lnc < 0] = 0
            Lnc = Lnc - L1c
            Lnc[Lnc < 0] = 0
            Lnc = Lnc.A
            Lnn = Ln.A * Lnc

            print(Lnn)
            L_list.append(np.copy(Lnn))
            L0, L1 = np.matrix(np.copy(L1)), np.matrix(np.copy(Ln))
            L0_, L1_ = np.matrix(np.copy(L1_)), np.matrix(np.copy(Ln_))
        print(L_list)
        return np.concatenate(L_list, axis=-1)
    elif K == 1:
        return np.asarray(L0)
    else:
        raise ValueError(f'ERROR: the size of spatial kernel must be greater than 1, but received "{K}".')

#original adjacency of graph G
A = np.array([[0.0, 1, 0, 0, 0],
                [1, 0, 1, 1, 1],
                [0, 1, 0, 0, 0],
                [0, 1, 0, 0, 1],
                [0, 1, 0, 1, 0]])
#compute laplacian matrix
L = laplacian(A)
#the convolution kernel was obtained by Chebyshev polynomial
Lk = ms_kernel(A, 4, L.shape[0])