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
import numba
from numba import cuda, float64, int32
import scipy
from math import *
import sellcs

################
# CUDA kernels #
################
@cuda.jit
def cu_restrict(ipart: int32[:,:], part_size: int32[:], x: float64[:], x_c: float64[:]):
    r'''
    Restrict vector x to coarse vector x_c using a weighted sum (average) over partitions.
    Each thread block handles one partition.

    Args:
        ipart (int32[:,:]): 2D array where ipart[i, j] is the j-th global index in partition i.
                           Unused entries should be -1.
        part_size (int32[:]): 1D array where part_size[i] is the number of elements in partition i.
        x (float64[:]): Fine-level input vector on GPU.
        x_c (float64[:]): Coarse-level output vector on GPU.

    Launch Configuration:
        Blocks: nparts (one block per coarse element)
        Threads: 128 (must match s_mem size)
    '''
    # Allocate shared memory for the reduction
    # Size should match the number of threads per block
    s_mem = cuda.shared.array(shape=128, dtype=float64)

    # Each block handles exactly one partition
    part_idx = cuda.blockIdx.x
    if part_idx >= x_c.size:
        return

    tid = cuda.threadIdx.x
    stride = cuda.blockDim.x
    n_elements = part_size[part_idx]

    # 1. Grid-stride loop within the block to sum all elements in this partition
    local_sum = 0.0
    for j in range(tid, n_elements, stride):
        idx = ipart[part_idx, j]
        local_sum += x[idx]

    s_mem[tid] = local_sum
    cuda.syncthreads()

    # 2. In-block reduction (Tree reduction)
    # This assumes blockDim.x is a power of 2 (e.g., 128)
    i = cuda.blockDim.x // 2
    while i > 0:
        if tid < i:
            s_mem[tid] += s_mem[tid + i]
        cuda.syncthreads()
        i //= 2

    # 3. Write result to global memory
    if tid == 0:
        # Final coarsening: Average of the sums
        if n_elements > 0:
            x_c[part_idx] = s_mem[0] / n_elements
        else:
            x_c[part_idx] = 0.0


@cuda.jit
def cu_prolongate(part: int32[:], x_c: float64[:], x: float64[:]):
    r'''
    Prolongate coarse vector x_c to fine vector x by broadcasting coarse values to all
    fine indices in the corresponding partition.

    Args:
        part (int32[:]): Mapping from fine index i to its partition ID.
        x_c (float64[:]): Coarse-level input vector on GPU.
        x (float64[:]): Fine-level output vector on GPU.

    Launch Configuration:
        Total threads should be at least N (fine vector size).
    '''
    idx = cuda.grid(1)
    if idx < x.size:
        x[idx] = x_c[part[idx]]


@cuda.jit
def cu_csr_project_atomic(values, col_indices, row_ptr, row_map, col_map, consts, C):
    r'''
    Compute a weighted projection of a CSR matrix into a dense matrix C using atomic additions.
    This is used for constructing the coarse-grid operator E = V^T A V.

    Args:
        values (float64[:]): CSR matrix data.
        col_indices (int32[:]): CSR column indices.
        row_ptr (int32[:]): CSR row pointers.
        row_map (int32[:]): Mapping from CSR row to dense row in C (-1 if excluded).
        col_map (int32[:]): Mapping from CSR column to dense column in C (-1 if excluded).
        consts (float64[:]): Weights for each target row/column (e.g., 1/n_members).
        C (float64[:,:]): Dense output matrix on GPU.
    '''
    row_idx = cuda.grid(1)
    
    if row_idx < row_ptr.shape[0] - 1:
        # Determine which row of C this row of A maps to
        target_row = row_map[row_idx]
        if target_row == -1: return # Skip if row not in deflation space
        
        c_i = consts[target_row]
        
        # Iterate over non-zeros in CSR row
        for nz_idx in range(row_ptr[row_idx], row_ptr[row_idx + 1]):
            col_idx = col_indices[nz_idx]
            val = values[nz_idx]
            
            # Determine which column of C this entry maps to
            target_col = col_map[col_idx]
            if target_col == -1: continue
            
            c_l = consts[target_col]
            
            # Weighted contribution to the dense matrix
            # Note: AtomicAdd is necessary because multiple threads 
            # contribute to the same C[target_row, target_col]
            contribution = val * c_i * c_l
            cuda.atomic.add(C, (target_row, target_col), contribution)
