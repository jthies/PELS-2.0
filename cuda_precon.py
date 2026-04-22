import math
import numpy as np
import numba
from numba import cuda, float64

@cuda.jit
def cu_invert(v: float64[:]):
    idx = cuda.grid(1)
    if idx < v.size:
        v[idx] = 1.0/v[idx]

@cuda.jit
def cu_ichol0(L_data, L_row_idx, L_col_idx, L_indptr):
    '''
    CUDA implementation of an asynchronous incomplete Cholesky-0
    factorization.

    Input: L=tril(A).tocsr(), with (L_data, L_col_idx, L_indptr) the CSR arrays (obtained from to_device(L)),
           and (L_row_idx, L_col_idx) the (i,j) arrays, obtained from to_device(L.tocoo().[row,col])
    This kernels is to be launched with at least L.nnz threads.
    '''
    # Map each thread to an element in the sparse data array
    idx = cuda.grid(1)

    if idx >= L_data.size:
        return

    # Get the original value from matrix A before overwriting it.
    # We need it for our fix-point iteration below.
    aij = L_data[idx]

    # row and column index (i,j) for this thread:
    row_i = L_row_idx[idx]
    col_j = L_col_idx[idx]

    # start and end of rows i and j
    start_i = L_indptr[row_i]
    end_i = L_indptr[row_i + 1]
    start_j = L_indptr[col_j]
    end_j = L_indptr[col_j + 1]

    # We need to find the location of the diagonal element L_jj
    # Most likely, it's the last entry in every row, but we don't know that for sure.
    diag_j = -1
    for d in range(end_j-1, start_j-1, -1):
        if L_col_idx[d] == col_j:
            diag_j = d
            break
    if diag_j == -1:
        raise Exception('missing diagonal element in L')

    # Asynchronous iteration: Based on the current L,
    # compute sum_k(L_ik * L_jk). The elements L_ik, Ljk)
    # may be updated by other threads while we're at it, but
    # the method should converge for spd matrices.
    # We allow a maximum number of 10 sweeps per element,
    # and stop if the update becomes very small
    maxit = 5
    convtol = abs(aij)*1e-2
    for _ in range(maxit):

        sum_lk = 0.0

        # 'sparse dot product':
        curr_i = start_i
        curr_j = start_j

        while curr_i < end_i and curr_j < end_j:
            k_i = L_col_idx[curr_i]
            k_j = L_col_idx[curr_j]

            # We can exploit that i>=j (we're only working on a lower triangular matrix).
            if k_i < col_j and k_j < col_j:
                if k_i == k_j:
                    sum_lk += L_data[curr_i] * L_data[curr_j]
                    curr_i += 1
                    curr_j += 1
                elif k_i < k_j:
                    curr_i += 1
                else:
                    curr_j += 1
            else:
                break

        # Update L_data[idx] based on whether it is a diagonal or off-diagonal
        L_ij_old = L_data[idx]
        if row_i == col_j:
            # Diagonal: L_ii = sqrt(a_ii - sum)
            # This square-root is OK because spd matrices are diagonally dominant:
            L_data[idx] = math.sqrt(aij - sum_lk)
        else:
            # Off-diagonal: L_ij = (a_ij - sum) / L_jj
            L_data[idx] = (aij - sum_lk) / L_data[diag_j]

        if abs(L_ij_old-L_data[idx])<convtol:
            break
