import numpy
import scipy
import scipy.sparse.linalg as spla
from math import sqrt
import pymetis
from time import perf_counter

def partition_sparse_matrix(matrix, nparts):
    # 2. Extract adjacency info from CSR format
    # indptr tells us where each row's data starts and ends
    # indices contains the actual column neighbors
    adjacency_list = [
        matrix.indices[matrix.indptr[i] : matrix.indptr[i+1]]
        for i in range(matrix.shape[0])
    ]

    # 3. Perform the partition
    # ncuts is the number of edges that span across different partitions
    # membership is an array mapping each node to a partition ID
    ncuts, part = pymetis.part_graph(nparts, adjacency=adjacency_list)

    return part


class DeflatedOperator:

    def __init__(self, A, nc):
        '''
        Construct deflated operator with given coarse problem size (nc x nc).
        We use pymetis to partition the graph into nc parts, and set 
        V(i,p)=1 (if node i is in partition p), and define the subspace
        as normalize(V).
        '''

        if A.shape[0] != A.shape[1]:
            raise Exception('A must be square!')

        self.A = A
        self.dtype = A.dtype
        self.shape = A.shape
        self.nc = nc
        #print('partition the matrix using METIS...')
        t0 = perf_counter()
        self.part = partition_sparse_matrix(A, nc)
        t1 = perf_counter()
        #print(f'elapsed time: {t1-t0:4.2g}s')

        n = A.shape[0]
        # determine the size of each partition
        p, n_members = numpy.unique(self.part, return_counts=True)

        valV  = 1.0 / n_members[self.part]
        rows = numpy.arange(n,dtype='int32')
        cols = self.part

        self.V = scipy.sparse.csc_matrix( (valV, (rows, cols)), shape=[n,nc])
        self.E = scipy.sparse.csc_matrix(self.V.T @ (self.A @ self.V))
        self.Elu = spla.splu(self.E)

    def applyQ(self, x):
        '''
        computes y = Q*x
        with Q = V(V'AV)\V'
        '''
        q = self.V.T @ x
        return self.V @ self.Elu.solve(q)

    def proj(self, x):
        '''
        computes y = (I - QA)x
        with Q = V(V'AV)\V'
        '''
        return x - self.applyQ(self.A @ x)

    def projT(self, x):
        '''
        computes y = (I - AQ)x
        with Q = V(V'AV)\V'
        '''
        return x - self.A @ self.applyQ(x)

    def _matvec(self, x):
        '''
        apply deflated operator A,
          y = (I - AQ)A(I - QA)x,
          Q = V (V'AV)\(V'x)
        '''
        y = self.projT(self.A @ self.proj(x))
        return y

