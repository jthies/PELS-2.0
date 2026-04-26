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
from cuda_precon import *

from cupy_kernels import as_cupy
from cupyx.scipy.sparse.linalg import spilu

def compile_all():
    n=10
    x=to_device(np.ones(n,dtype='float64'))
    cu_invert.forall(n)(x)

    A =(scipy.sparse.rand(n,n,0.6) + scipy.sparse.eye(n,n)).tocsr()
    L =to_device(scipy.sparse.tril(A).tocsr())
    cu_rows = copy(L.cu_indices)
    cu_ichol_pre.forall(n)(L.cu_data,L.cu_indptr, cu_rows, L.cu_indices)
    cu_ichol0.forall(L.nnz)(L.cu_data, L.cu_indptr, cu_rows, L.cu_indices)

# total number of calls
calls = {'setup': 0, 'apply': 0}
# total elapsed time in seconds
time = {'setup': 0, 'apply': 0.0, 'axpby': 0.0}


class Jacobi:
    '''
    The most basic preconditioner imaginable: M=diag(A)
    '''
    def __init__(self, A):
        t0 = perf_counter()
        self.D_inv = to_device(A.diagonal())
        cu_invert.forall(self.D_inv.size)(self.D_inv)
        cuda.synchronize()
        t1 = perf_counter()
        calls['setup'] += 1
        time['setup'] += t1-t0

    def apply(self, w, v):
        '''
        Diagonal scaling, v = D^{-1}w
        '''
        t0 = perf_counter()
        cu_vscale.forall(v.size)(self.D_inv, w, v)
        cuda.synchronize()
        t1 = perf_counter()
        calls['apply'] += 1
        time['apply'] += t1-t0

class SymmetricGaussSeidel:
    '''
    The most basic preconditioner imaginable: M=diag(A)
    '''
    def __init__(self, A):
        t0 = perf_counter()
        self.D = to_device(A.diagonal())
        self.LplusD = to_device(scipy.sparse.tril(A).tocsr())
        self.v_tmp = to_device(np.zeros(A.shape[0]))
        t1 = perf_counter()
        calls['setup'] += 1
        time['setup'] += t1-t0

    def apply(self, w, v):
        '''
        Symmetric Gauss-Seidel: v = (L+D)^{-T} D (L+D)^{-1} w
        '''
        t0 = perf_counter()
        trsv(self.LplusD, w, self.v_tmp)
        cu_vscale_inplace.forall(v.size)(self.D, self.v_tmp)
        cuda.synchronize()
        trsv(self.LplusD, self.v_tmp, v, transpose=True)
        t1 = perf_counter()
        calls['apply'] += 1
        time['apply'] += t1-t0

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
        t0 = perf_counter()
        self.shape = A.shape
        self.dtype = A.dtype
        # create a temporary vector for the 'apply' function:
        self.v_tmp = to_device(np.zeros(A.shape[0]))
        # Use the lower triangular factor of A as an initial guess:
        self.L = to_device(scipy.sparse.tril(A).tocsr())
        # our simple IC0 kernel requires the diagonal element to be stored as the
        # last element in each row.
        data = self.L.cu_data
        col_idx = self.L.cu_indices
        row_idx = cuda.device_array_like(col_idx)
        indptr = self.L.cu_indptr
        cu_ichol_pre.forall(self.shape[0])(data, indptr, row_idx, col_idx)
        cuda.synchronize()
        cu_ichol0.forall(self.L.nnz)(data, row_idx, col_idx, indptr)
        cuda.synchronize()
        t1 = perf_counter()
        calls['setup'] += 1
        time['setup'] += t1-t0
        # for debugging
        # CholFactor = from_device(self.L)
        # scipy.io.mmwrite('CholFactor',CholFactor)


    def apply(self, w, v):
        '''
        Apply the preconditioner M^{-1} to a vector, i.e.,
        Mv = w -> LL^Tv = w
               -> v = L^{-T} L^{-1} w
        '''
        t0 = perf_counter()
        if not (cuda.is_cuda_array(v) and
                cuda.is_cuda_array(w)):
            raise Exception('IChol0 preconditioner requires vectors to be cuda arrays')

        trsv(self.L, w, self.v_tmp, False)
        trsv(self.L, self.v_tmp, v, True)
        t1 = perf_counter()
        calls['apply'] += 1
        time['apply'] += t1-t0

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
        t0 = perf_counter()
        self.shape = A.shape
        self.dtype = A.dtype
        self.A = to_device(scipy.sparse.tril(A).tocsr())
        self.ilu0 = spilu(as_cupy(A), drop_tol=None, fill_factor=1)
        t1 = perf_counter()
        calls['setup'] += 1
        time['setup'] += t1-t0


    def apply(self, w, v):
        t0 = perf_counter()
        if not (cuda.is_cuda_array(v) and
                cuda.is_cuda_array(w)):
            raise Exception('IChol0 preconditioner requires vectors to be cuda arrays')

        v[:] = self.ilu0.solve(as_cupy(w))
        t1 = perf_counter()
        calls['apply'] += 1
        time['apply'] += t1-t0

import pyamg

class PyAMG:

    def __init__(self, A):
        t0 = perf_counter()
        self.A  = from_device(A)
        self.AMG = pyamg.ruge_stuben_solver(self.A, max_coarse=20, max_levels=10)
        self.M = self.AMG.aspreconditioner(cycle='V')
        t1 = perf_counter()
        calls['setup'] += 1
        time['setup'] += t1-t0

    def apply(self, w, v):
        t0 = perf_counter()
        v[:] = self.M@from_device(w)
        t1 = perf_counter()
        calls['apply'] += 1
        time['apply'] += t1-t0

