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
import scipy.sparse.linalg as spla
from time import perf_counter
from numba import cuda
import pymetis

from kernels import to_device, clone, axpby, spmv
from cuda_deflation import cu_restrict, cu_prolongate

# total number of calls
calls = {'setup': 0, 'apply': 0}
# total elapsed time in seconds
time = {'setup': 0, 'apply': 0.0}

def reset_counters():
    ''' Reset performance counters for deflation setup and application. '''
    calls['setup'] = 0
    calls['apply'] = 0
    time['setup'] = 0.0
    time['apply'] = 0.0

def perf_report():
    ''' Print a performance report for the deflation operator. '''
    print('Deflation report:')
    print('  setup calls: %d, time: %10.4g s'%(calls['setup'], time['setup']))
    print('  apply calls: %d, time: %10.4g s'%(calls['apply'], time['apply']))

def compile_all():
    ''' 
    Pre-compile JIT kernels and class methods by running a small problem. 
    This removes compilation overhead from subsequent benchmark runs.
    '''
    n = 128
    nc = 4
    # Create a small SPD matrix for compilation
    A_csr = scipy.sparse.diags([[-1]*n, [2]*n, [-1]*n], [-1, 0, 1]).tocsr()
    A_gpu = to_device(A_csr)
    x = to_device(np.ones(n, dtype='float64'))
    y = to_device(np.zeros(n, dtype='float64'))

    A_defl = DeflatedOperator(A_csr, A_gpu, nc)
    A_defl.applyQ(x, y)
    A_defl.proj(x, y)
    reset_counters()

def partition_sparse_matrix(matrix, nparts):
    '''
    Partition a sparse matrix's graph into nparts using METIS.

    Args:
        matrix: A scipy.sparse.csr_matrix.
        nparts (int): Number of partitions.

    Returns:
        np.array (int32): Partition ID for each row/node.
    '''
    # Extract adjacency info from CSR format
    adjacency_list = [
        matrix.indices[matrix.indptr[i] : matrix.indptr[i+1]]
        for i in range(matrix.shape[0])
    ]

    # Perform the partition
    ncuts, part = pymetis.part_graph(nparts, adjacency=adjacency_list)

    return np.array(part, dtype='int32')

class DeflatedOperator:
    '''
    Implements the Deflated Conjugate Gradient (DPCG) operator.
    The subspace is defined by a partition of the domain (coarse grid).
    Q = V (V^T A V)^{-1} V^T is the projection into the deflation space.
    '''
    def __init__(self, A_csr, A_gpu, nc):
        '''
        Initialize the deflated operator by partitioning the matrix and 
        factorizing the coarse-grid operator E = V^T A V.

        Args:
            A_csr (scipy.sparse.csr_matrix): System matrix on CPU for setup.
            A_gpu: System matrix on GPU for projection operations.
            nc (int): Number of coarse partitions (size of coarse problem).
        '''
        t0 = perf_counter()
        self.A_gpu = A_gpu
        self.nc = nc
        self.n = A_csr.shape[0]

        # 1. Partitioning using METIS
        self.part = partition_sparse_matrix(A_csr, nc)
        self.cu_part = cuda.to_device(self.part)

        # 2. Prepare ipart and part_size for GPU restriction (cu_restrict)
        # ipart[nc, max_part_size] maps partition ID and local index to global index.
        p, n_members = np.unique(self.part, return_counts=True)

        full_n_members = np.zeros(nc, dtype='int32')
        full_n_members[p] = n_members.astype('int32')
        n_members = full_n_members

        max_part_size = np.max(n_members)
        self.part_size = n_members.astype('int32')
        self.cu_part_size = cuda.to_device(self.part_size)

        self.ipart = np.full((nc, max_part_size), -1, dtype='int32')
        counters = np.zeros(nc, dtype='int32')
        for i, p_id in enumerate(self.part):
            self.ipart[p_id, counters[p_id]] = i
            counters[p_id] += 1
        self.cu_ipart = cuda.to_device(self.ipart)

        # 3. Construct Coarse matrix E = V^T A V on CPU
        # V is n x nc, V[i, p] = 1/n_members[p] if part[i] == p.
        rows = np.arange(self.n)
        cols = self.part
        safe_n_members = np.where(n_members == 0, 1, n_members)
        vals = 1.0 / safe_n_members[self.part]
        V = scipy.sparse.csc_matrix((vals, (rows, cols)), shape=(self.n, nc))
        self.V = V # Save for debug if needed
        self.E = V.T @ (A_csr @ V)

        # 4. Factorize E on CPU using SuperLU
        self.Elu = spla.splu(self.E)

        # 5. Allocate buffers for restriction/prolongation on GPU
        self.q_c = cuda.device_array(nc, dtype='float64')
        self.y_c_gpu = cuda.device_array(nc, dtype='float64')

        t1 = perf_counter()
        calls['setup'] += 1
        time['setup'] += t1 - t0

    def applyQ(self, x, y):
        r''' 
        Compute y = Q*x = V (V^T A V)^{-1} V^T x.

        Args:
            x: Input vector on GPU.
            y: Output vector on GPU.
        '''
        t0 = perf_counter()

        # 1. Restriction: q_c = V' x (Weighted sum per partition)
        threadsperblock = 128
        blockspergrid = self.nc
        cu_restrict[blockspergrid, threadsperblock](self.cu_ipart, self.cu_part_size, x, self.q_c)

        # 2. Coarse solve on CPU
        q_c_host = self.q_c.copy_to_host()
        y_c_host = self.Elu.solve(q_c_host)
        self.y_c_gpu.copy_to_device(y_c_host)

        # 3. Prolongation: y = V y_c (Broadcast coarse values)
        threadsperblock = 256
        blockspergrid = (self.n + threadsperblock - 1) // threadsperblock
        cu_prolongate[blockspergrid, threadsperblock](self.cu_part, self.y_c_gpu, y)

        cuda.synchronize()
        t1 = perf_counter()
        calls['apply'] += 1
        time['apply'] += t1 - t0

    def proj(self, x, y):
        r''' 
        Compute the projection orthogonal to the deflation space: y = (I - QA)x.

        Args:
            x: Input vector on GPU.
            y: Output vector on GPU.
        '''
        # We need a temporary for A@x
        tmp = clone(x)
        spmv(self.A_gpu, x, tmp)

        # y = Q (A x)
        self.applyQ(tmp, y)

        # y = x - y
        axpby(1.0, x, -1.0, y)
