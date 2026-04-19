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
from kernels import *

def cg_solve(A, M, b, x0, tol, maxit, verbose=True, x_ex=None):
    '''
    Right-Preconditioned CG Solver
    A: System matrix
    M: Preconditioner (must have an .apply(in, out) or similar method)
    '''
    x = clone(b)
    r = clone(b)
    p = clone(b)
    q = clone(b)

    if M is None:
        z = r
    else:
        z = clone(b) # Preconditioned residual

    if x_ex is not None:
        print('PerfWarning: Extra operations enabled for error norm.')
        err = clone(b)

    tol2 = tol*tol

    # Initial Residual: r = b - A*x0
    axpby(1.0, x0, 0.0, x)
    if hasattr(A, 'apply'):
        A.apply(x, r)
    else:
        spmv(A, x, r)
    axpby(1.0, b, -1.0, r)

    # Initial Preconditioning: z = M^-1 * r
    if M is not None:
        M.apply(r, z)

    # Initial search direction: p = z
    axpby(1.0, z, 0.0, p)

    # rho = <r, z> (The preconditioned scalar)
    rho = dot(r, z)

    # We use the true residual norm for the stopping criterion
    if M is not None:
        res_norm_sq = dot(r, r)
    else:
        res_norm_sq = rho

    if verbose:
        print('%d\t%e'%(0, np.sqrt(res_norm_sq)))

    for iter in range(maxit + 1):

        # Check stopping criteria on the true residual
        if res_norm_sq < tol2:
            break

        # q = A * p
        if hasattr(A, 'apply'):
            A.apply(p, q)
        else:
            spmv(A, p, q)

        # alpha = <r, z> / <p, Ap>
        pq = dot(p, q)
        alpha = rho / pq

        # x = x + alpha * p
        axpby(alpha, p, 1.0, x)

        # r = r - alpha * q
        axpby(-alpha, q, 1.0, r)

        # Update preconditioned residual: z = M^-1 * r
        if M is not None:
            M.apply(r, z)

        rho_old = rho
        rho = dot(r, z)
        if M is not None:
            res_norm_sq = dot(r, r)
        else:
            res_norm_sq = rho

        if verbose:
            if x_ex is not None:
                axpby(1.0, x, 0.0, err)
                axpby(-1.0, x_ex, 1.0, err)
                err_norm = np.sqrt(dot(err, err))
                print('%d\t%e\t%e'%(iter+1, np.sqrt(res_norm_sq), err_norm))
            else:
                print('%d\t%e'%(iter+1, np.sqrt(rho)))

        # beta = <r_new, z_new> / <r_old, z_old>
        beta = rho / rho_old

        # p = z + beta * p
        axpby(1.0, z, beta, p)

    res_norm_sq = dot(r, r)
    return x, np.sqrt(res_norm_sq), iter

if __name__ == '__main__':

    import numba
    import gc
    from numpy.linalg import norm
    from scipy.sparse import *
    from scipy.io import mmread
    from sellcs import sellcs_matrix

    from matrix_generator import create_matrix
    from pels_args import *

    ## **Note:** The Python garbage collector (gc)
    ## can kill the performance of the C kernels
    ## for some obscure reason (possibly a conflict
    ## between Numba/LLVM and other compilers like GCC).
    ## For the pure Python/numba/cuda kernels, this is not
    ## the case, but if you are facing obvious performnace
    ## problems with the C kernels, you may want to disalbe
    ## garbage collection:
    gc.disable()

    parser = get_argparser()

    # add driver-specific command-line arguments for polynomial preconditioning with or without RACE:
    parser.add_argument('-printerr', action=BooleanOptionalAction,
                    help='Besides the residual norm, also compute and print the error norm.')
    parser.add_argument('-rhsfile', type=str, default='None',
                    help='MatrixMarket filename for right-hand side vector b')
    parser.add_argument('-solfile', type=str, default='None',
                    help='MatrixMarket filename for exact solution x')
    parser.add_argument('-poly_k', type=int, default=0,
                    help='Use a degree-k polynomial preconditioner based on the Neumann series.')
    parser.add_argument('-use_RACE', action=BooleanOptionalAction,
                    help='Use RACE for cache blocking.')
    parser.add_argument('-cache_size', type=float, default=30,
                    help='Cache size used to perform RACE\'s cache blocking')


    args = parser.parse_args()

    if args.seed is not None:
        np.random.seed(args.seed)

    if args.matfile != 'None':
        if args.matgen!='None':
            print('got both -matfile and -matgen, the latter will be ignored.')
        if not args.matfile.endswith(('.mm','.mtx','.mm.gz','.mtx.gz')):
            raise(ValueError('Expecting MatrixMarket file with extension .mm[.gz] or .mtx[.gz]'))
        A = csr_matrix(mmread(args.matfile))
    elif args.matgen != 'None':
        A = create_matrix(args.matgen)
    N = A.shape[0]

    if args.solfile!='None':
        x_ex=mmread(args.solfile).reshape(N)
    else:
        x_ex=np.random.rand(N)

    if args.rhsfile!='None':
        b=mmread(args.rhsfile).reshape(N)
    else:
        b=A*x_ex

    x0 = np.zeros(N,dtype='float64')

    print('norm of rhs: %e'%(norm(b)))
    print('rel. residual of given solution: %e'%(norm(A*x_ex-b)/norm(b)))

    tol = args.tol
    maxit = args.maxit

    sigma=1

    A_csr = A # we may need it for creating the preconditioner
              # in case the user wants a SELL-C-sigma matrix.

    if args.fmt=='SELL':
        C=args.C
        sigma=args.sigma
        A = sellcs_matrix(A_csr=A_csr, C=C, sigma=sigma)
        b = b[A.permute]
        print('Matrix format: SELL-%d-%d'%(C,sigma))
        A_csr = A_csr[A.permute[:,None], A.permute]
    else:
        print('Matrix format: CSR')

    M = None

    if available_gpus()>0:
        type = 'gpu'
    else:
        type = 'cpu'

    print('Will run on '+type)

    if type=='gpu':
        x0 = to_device(x0)
        b  = to_device(b)
        A  = to_device(A)

    # take compilation time out of the balance:
    compile_all()

    # we want to make sure what we measure during CG in total
    # is consistent with the sum of the kernel calls and their
    # runtime as predicted by the roofline model, so reset all
    # counters and timers:
    reset_counters()

    t0 = perf_counter()

    if M is not None:
        t0_pre = perf_counter()
        # setup preconditioner...
        # ...
        if args.fmt == 'SELL':
            # note: If A was originally sorted by row-length (sigma>1), use the same
            # sorting for L and U to avoid intermittent permutation by setting sigma=1.
            # There still seems to be some kind of bug, though, because the number of
            # iterations will increase with poly_k>0 and sigma>1. Hence this warning.
            M.L = to_device(sellcs_matrix(A_csr=M.L, C=args.C, sigma=1))
        t1_pre = perf_counter()

    x_ex_in = None
    if args.printerr:
        x_ex_in = x_ex

    t0_soln = perf_counter()
    x, relres, iter = cg_solve(A,M, b,x0,tol,maxit, x_ex=x_ex_in)
    t1_soln = perf_counter()

    t1 = perf_counter()
    t_CG = t1-t0

    if M is not None:
        t_pre = t1_pre-t0_pre
        t_soln = t1_soln-t0_soln

    x = to_host(x)

    print('number of CG iterations: %d'%(iter))
    res = np.empty_like(x)
    spmv(A_csr,x,res)
    res=b-res
    print('relative residual of computed solution: %e'%(norm(res)/norm(b)))

    if args.fmt=='SELL' and sigma>1:
        x = x[A.unpermute]

    print('relative error of computed solution: %e'%(norm(x-x_ex)/norm(x_ex)))

    hw_string = type
    if type=='cpu':
        hw_string+=' ('+str(numba.threading_layer())+', '+str(numba.get_num_threads())+' threads)'
    print('Hardware: '+hw_string)
    perf_report(type)
    if M is not None:
        print('Total time for constructing precon: %g seconds.'%(t_pre))
        print('Total time for solving: %g seconds.'%(t_soln))
    print('Total time for CG: %g seconds.'%(t_CG))

