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
    Restrict vector x to coarse vector x_c.

    Input:   ipart[nparts, max(part_size)]: row-major 2D array s.t. ipart[i,j] is the j'th index in partition i.
             part_size[nparts]: number of indices in each partition.
             X[N]: CUDA array representing a 'fine' vector to be 'coarsened'.
    Output:  x_c[nparts], CUDA array with x_c[i] = (sum_{part==i} x)//part_size[i]

    This kernel should be launched with at least nparts thread blocks of exactly 128 threads each.
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
    # This assumes blockDim.x is a power of 2 (e.g., 256)
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
    Prolongate coarse vector x_c to vector x.

    Input:   part[N]: part[i] indicates to which partition (coarse index) a fine index belongs
             x_c[nparts], CUDA array with one element per partition
    Output   x[N]: CUDA array representing a 'fine' vector

    This kernel should be launched with at least N threads.
    '''
    idx = cuda.grid(1)
    x[idx] = x_c[part[idx]]


@cuda.jit
def cu_csr_project_atomic(values, col_indices, row_ptr, row_map, col_map, consts, C):
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


