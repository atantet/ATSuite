import numpy as np
from scipy import sparse

def toRightStochastic(M, fill='none'):
    N = M.shape[0]
    P = np.zeros((N, N))
    for ii in range(N):
        norm = np.sum(M[ii])
        if norm != 0:
            P[ii] = M[ii] * 1. / norm
    return P


def toLeftStochastic(M, fill='none'):
    N = M.shape[0]
    P = np.zeros((N, N))
    for ii in range(N):
        norm = np.sum(M[:, ii])
        if norm != 0:
            P[:, ii] = M[:, ii] * 1. / norm
    return P


def toRightStochasticCSR(csr_mat):
    N = csr_mat.shape[0]
    T = csr_mat.copy().astype(float)
    for k in range(N):
        norm = T[k].sum()
        if norm != 0:
            T.data[T.indptr[k]:T.indptr[k+1]] /= norm
    return T


