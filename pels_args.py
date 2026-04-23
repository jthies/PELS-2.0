from argparse import *

def get_argparser():
    '''
    pels_argparser() returns an argparse.ArgumentParser object that
    offers some useful settings for, e.g.,:

    - selecting the sparse matrix format (CSR or SELL-C-sigma)
    - setting solver parameters like number of iterations and convergence tolerance
    - etc.

    For a full list, run your driver with the --help optino.
    '''
    parser = ArgumentParser(description='Run a CG benchmark.')
    parser.add_argument('-matfile', type=str, default='None',
                    help='MatrixMarket filename for matrix A')
    parser.add_argument('-matgen', type=str, default='None',
                    help='Matrix generator string  for matrix A. E.g., "Laplace128x64", '+
                         '"Laplace50x50x50", or "LinElast100x50" (latter requires pyamg)')
    parser.add_argument('-maxit', type=int, default=1000,
                    help='Maximum number of CG iterations allowed.')
    parser.add_argument('-tol', type=float, default=1e-6,
                    help='Convergence criterion: ||b-A*x||_2/||b||_2<tol')
    parser.add_argument('-fmt', type=str, default='CSR',
                    help='Sparse matrix format to be used [CSR, SELL]')
    parser.add_argument('-C', type=int, default=1,
                    help='Chunk size C for SELL-C-sigma format.')
    parser.add_argument('-sigma', type=int, default=1,
                    help='Sorting scope sigma for SELL-C-sigma format.')
    parser.add_argument('-seed', type=int, default=None,
                    help='Random seed to make runs reproducible')
    parser.add_argument('-precon', type=str, default=None,
                    help='Preconditioner to be used [None,Jacobi,SGS,IC0]')

    return parser
