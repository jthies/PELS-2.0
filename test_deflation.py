#/*******************************************************************************************/
#/* This file is part of the training material available at                                 */
#/* https://github.com/jthies/PELS                                                          */
#/* You may redistribute it and/or modify it under the terms of the BSD-style licence       */
#/* included in this software.                                                              */
#/*                                                                                         */
#/* Contact: Jonas Thies (j.thies@tudelft.nl)                                               */
#/*                                                                                         */
#/*******************************************************************************************/

import unittest
import pytest
import numpy as np
import scipy.sparse
from kernels import to_device, to_host, clone, available_gpus
from deflation import DeflatedOperator, partition_sparse_matrix
from matrix_generator import create_matrix

class DeflationTest(unittest.TestCase):

    def setUp(self):
        np.random.seed(12345678)
        self.n = 200
        self.nc = 10
        # Create a simple 1D Laplace matrix
        self.A_csr = scipy.sparse.diags([[-1]*(self.n-1), [2]*self.n, [-1]*(self.n-1)], [-1, 0, 1]).tocsr()
        self.A_gpu = to_device(self.A_csr)
        
        # Exact solution and RHS
        self.x_ex = np.random.rand(self.n)
        self.b_gpu = to_device(self.A_csr @ self.x_ex)
        
        # Deflated Operator
        self.A_defl = DeflatedOperator(self.A_csr, self.A_gpu, self.nc)
        
        self.eps = 1e-12

    def test_partitioning(self):
        part = partition_sparse_matrix(self.A_csr, self.nc)
        assert len(part) == self.n
        assert np.max(part) == self.nc - 1
        assert np.min(part) == 0
        # Check that it returns an array of int32 as expected by CUDA kernels
        assert part.dtype == np.int32

    def test_applyQ_property(self):
        ''' Test that Q = V(V'AV)^{-1}V' satisfies V'A Q = V' '''
        x = to_device(np.random.rand(self.n))
        y = clone(x)
        self.A_defl.applyQ(x, y)
        
        y_host = to_host(y)
        x_host = to_host(x)
        
        # V' A y should match V' x
        lhs = self.A_defl.V.T @ (self.A_csr @ y_host)
        rhs = self.A_defl.V.T @ x_host
        
        error = np.linalg.norm(lhs - rhs) / np.linalg.norm(rhs)
        assert error < self.eps

    def test_applyQ_idempotent(self):
        ''' Test that Q A Q = Q '''
        x = to_device(np.random.rand(self.n))
        y1 = clone(x)
        y2 = clone(x)
        temp = clone(x)
        
        # y1 = Q x
        self.A_defl.applyQ(x, y1)
        
        # temp = A (Q x)
        import kernels
        kernels.spmv(self.A_gpu, y1, temp)
        
        # y2 = Q (A Q x)
        self.A_defl.applyQ(temp, y2)
        
        error = np.linalg.norm(to_host(y1) - to_host(y2)) / np.linalg.norm(to_host(y1))
        assert error < self.eps

    def test_proj_orthogonality(self):
        ''' Test that P = I - QA satisfies V' A P = 0 '''
        x = to_device(np.random.rand(self.n))
        y = clone(x)
        
        # y = (I - QA) x
        self.A_defl.proj(x, y)
        
        y_host = to_host(y)
        
        # V' A y should be zero
        res = self.A_defl.V.T @ (self.A_csr @ y_host)
        error = np.linalg.norm(res) / np.linalg.norm(to_host(x))
        assert error < self.eps

    def test_proj_idempotent(self):
        ''' Test that P = I - QA is idempotent: P^2 = P '''
        x = to_device(np.random.rand(self.n))
        y1 = clone(x)
        y2 = clone(x)
        
        # y1 = P x
        self.A_defl.proj(x, y1)
        
        # y2 = P (P x)
        self.A_defl.proj(y1, y2)
        
        error = np.linalg.norm(to_host(y1) - to_host(y2)) / np.linalg.norm(to_host(y1))
        assert error < self.eps

@pytest.mark.parametrize('Matrix', ['Ddiag13', 'Dtest33'])
def test_deflation_parametrized(Matrix):
    import scipy.io
    A_csr = scipy.sparse.csr_matrix(scipy.io.mmread('matrices/'+Matrix+'.mm.gz'))
    n = A_csr.shape[0]
    nc = min(n // 2, 5) # Small nc for small matrices
    
    A_gpu = to_device(A_csr)
    A_defl = DeflatedOperator(A_csr, A_gpu, nc)
    
    x = to_device(np.random.rand(n))
    y = clone(x)
    A_defl.proj(x, y)
    
    # Orthogonality check: V' A P x = 0
    y_host = to_host(y)
    res = A_defl.V.T @ (A_csr @ y_host)
    assert np.linalg.norm(res) / np.linalg.norm(to_host(x)) < 1e-10

if __name__ == '__main__':
    unittest.main()
