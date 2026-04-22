#/*******************************************************************************************/
#/* This file is part of the training material available at                                 */
#/* https://github.com/jthies/PELS                                                          */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/

import numpy as np
import scipy
from kernels import *

from numba import cuda
from cuda_precon import cu_ichol0, cu_invert

from cupy_kernels import as_cupy
from cupyx.scipy.sparse.linalg import spilu

class Jacobi:
    '''
    The most basic preconditioner imaginable: M=diag(A)
    '''
    def __init__(self, A):
        self.D_inv = to_device(A.diagonal())
        cu_invert.forall(self.D_inv.size)(self.D_inv)
        cuda.synchronize()

    def apply(self, w, v):
        '''
        Diagonal scaling, v = D^{-1}w
        '''
        cu_vscale.forall(v.size)(self.D_inv, w, v)
        cuda.synchronize()

class SymmetricGaussSeidel:
    '''
    The most basic preconditioner imaginable: M=diag(A)
    '''
    def __init__(self, A):
        self.D = to_device(A.diagonal())
        self.LplusD = to_device(scipy.sparse.tril(A).tocsr())
        self.v_tmp = to_device(np.zeros(A.shape[0]))

    def apply(self, w, v):
        '''
        Symmetric Gauss-Seidel: v = (L+D)^{-T} D (L+D)^{-1} w
        '''
        trsv(self.LplusD, w, self.v_tmp)
        cu_vscale_inplace.forall(v.size)(self.D, self.v_tmp)
        cuda.synchronize()
        trsv(self.LplusD, self.v_tmp, v, transpose=True)

class IChol0:
    '''
    Zero-fill incomplete Cholesky factorization preconditioner.
    Given a symmetric and positive definite (spd) matrix A,
    computes A \approx LL^T, where L inherits the sparsity pattern
    of the lower triangular factor of A. The preconditioner is applied
    as a sequence of a forward triangular solve with L and a backward triangular
    solve with L^T.

    Example:

    A  = matrix_generator.create_matrix('Laplace128x128')
    IC = IChol0(A)
    y  = kernels.to_device(numpy.random.random(A.shape[0]))
    x  = kernels.to_device(numpy.zeros(A.shape[0]))
    IC.apply(y, x)
    '''

    def __init__(self, A):
        '''
        '''
        self.shape = A.shape
        self.dtype = A.dtype
        # create a temporary vector for the 'apply' function:
        self.v_tmp = to_device(np.zeros(A.shape[0]))
        # Use the lower triangular factor of A as an initial guess:
        self.L = to_device(scipy.sparse.tril(A).tocsr())
        L_coo = self.L.tocoo()
        data = self.L.cu_data
        row_idx = to_device(L_coo.row)
        col_idx = self.L.cu_indices
        indptr = self.L.cu_indptr
        cu_ichol0.forall(self.L.nnz)(data, row_idx, col_idx, indptr)
        cuda.synchronize()


    def apply(self, w, v):
        '''
        Apply the preconditioner M^{-1} to a vector, i.e.,
        Mv = w -> LL^Tv = w
               -> v = L^{-T} L^{-1} w
        '''
        if not (cuda.is_cuda_array(v) and
                cuda.is_cuda_array(w)):
            raise Exception('IChol0 preconditioner requires vectors to be cuda arrays')

        trsv(self.L, w, self.v_tmp, False)
        trsv(self.L, self.v_tmp, v, True)

class CuPyILU0:
    '''
    Zero-fill incomplete Cholesky factorization preconditioner.
    Given a symmetric and positive definite (spd) matrix A,
    computes A \approx LL^T, where L inherits the sparsity pattern
    of the lower triangular factor of A. The preconditioner is applied
    as a sequence of a forward triangular solve with L and a backward triangular
    solve with L^T.

    Example:

    A  = matrix_generator.create_matrix('Laplace128x128')
    IC = IChol0(A)
    y  = kernels.to_device(numpy.random.random(A.shape[0]))
    x  = kernels.to_device(numpy.zeros(A.shape[0]))
    IC.apply(y, x)
    '''

    def __init__(self, A):
        '''
        '''
        self.shape = A.shape
        self.dtype = A.dtype
        self.A = to_device(scipy.sparse.tril(A).tocsr())
        self.ilu0 = spilu(as_cupy(A), drop_tol=None, fill_factor=1)


    def apply(self, w, v):
        if not (cuda.is_cuda_array(v) and
                cuda.is_cuda_array(w)):
            raise Exception('IChol0 preconditioner requires vectors to be cuda arrays')

        v[:] = self.ilu0.solve(as_cupy(w))
