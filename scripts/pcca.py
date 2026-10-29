"""Robust PCCA+ (Deuflhard & Weber 2005; Roeblitz & Weber 2013): fuzzy metastable sets of a reversible Markov chain.

For a symmetric affinity W (sparse), the chain is P = D^-1 W with stationary distribution pi ~ D. Its k dominant
right eigenvectors X (first column constant, pi-orthonormal) are mapped by a linear transform A onto memberships
chi = X A: nonnegative, rows summing to one, each column a fuzzy set. A starts from the inner-simplex algorithm
(the k rows of X farthest apart) and is optimised for crispness, sum_j sum_i A_ij^2 / A_0j, which reaches k when
the sets are crisp; crispness / k near 1 says the chain really has k metastable sets.

    chi, crisp, vals = pcca(W, k)
    scan = pcca_scan(W, range(2, 16))   # {k: (chi, crispness / k)}, eigenvalues
"""

import numpy as np
import scipy.sparse as sp
from scipy.optimize import fmin
from scipy.sparse.linalg import eigsh


def dominant(W, k):
    """k dominant right eigenvectors of P = D^-1 W, pi-orthonormal, first one constant; and their eigenvalues."""
    d = np.asarray(W.sum(axis=1)).ravel()
    Dm = sp.diags(1 / np.sqrt(d))
    vals, V = eigsh((Dm @ W @ Dm).tocsr(), k=k, which="LA", tol=1e-6,
                    ncv=min(W.shape[0] - 1, max(4 * k, 2 * k + 1)))
    order = np.argsort(-vals)
    vals, V = vals[order], V[:, order]
    X = V / np.sqrt(d)[:, None] * np.sqrt(d.sum())
    X[:, 0] = np.abs(X[:, 0])  # the constant eigenvector: make it exactly one
    X[:, 0] = 1.0
    return X, vals


def inner_simplex(X, k):
    """Indices of k rows of X spanning the largest simplex (greedy), and the transform A0 = X[ind]^-1."""
    c = X[:, :k]
    ortho = c.copy()
    ind = np.zeros(k, int)
    ind[0] = int(np.argmax(np.linalg.norm(c, axis=1)))
    ortho -= c[ind[0]]
    for j in range(1, k):
        temp = ortho[ind[j - 1]].copy()
        temp /= max(np.linalg.norm(temp), 1e-300)
        ortho -= np.outer(ortho @ temp, temp)
        dist = np.linalg.norm(ortho, axis=1)
        dist[ind[:j]] = -1
        ind[j] = int(np.argmax(dist))
    return ind, np.linalg.inv(c[ind])


def _fill(crop, X):
    """Complete the lower-right block of A to a feasible transform (memberships >= 0, rows summing to one)."""
    x, y = crop.shape
    crop = np.concatenate([-crop.sum(1, keepdims=True), crop], axis=1)
    tmp = -(X[:, 1:] @ crop)
    top = tmp.max(axis=0, keepdims=True)
    A = np.concatenate([top, crop], axis=0)
    return A / top.sum()


def _crispness(A):
    return float(np.sum(A ** 2 / A[0][None, :]))


def pcca(W, k, X=None):
    """Memberships chi (n x k), crispness / k, and the transform A."""
    if X is None:
        X, _ = dominant(W, k)
    X = X[:, :k]
    _, A = inner_simplex(X, k)
    crop0 = A[1:, 1:].ravel()
    shape = A[1:, 1:].shape
    best = fmin(lambda v: -_crispness(_fill(v.reshape(shape), X)), crop0, disp=False, maxiter=4000 * len(crop0),
                maxfun=4000 * len(crop0))
    A = _fill(best.reshape(shape), X)
    chi = np.clip(X @ A, 0, 1)
    chi /= chi.sum(1, keepdims=True)
    return chi, _crispness(A) / k, A


def pcca_scan(W, ks):
    ks = list(ks)
    X, vals = dominant(W, max(ks) + 1)
    out = {}
    for k in ks:
        chi, crisp, _ = pcca(W, k, X)
        out[k] = (chi, crisp)
    return out, vals
