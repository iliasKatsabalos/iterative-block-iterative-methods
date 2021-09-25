import numpy as np
from numpy.core.defchararray import array
from scipy.linalg import norm, solve, block_diag
import time
from typing import *
import logging

def SOR(A: np.array, b: np.array, omega: np.float, precision: np.float = 10**-6, max_iter: int = 2000, 
    verbose: bool = False, return_iter: bool  = False) -> Union[np.array, int]:
    """Implements the Successive Overelaxation iterative algorithm and solves for Ax=b.
    Matrix A is broken down to two matrices. D is contains the diagonal elements and LU
    the non diagonal elements, such as A = D-L-U.

    Args:
        A (np.array): the matrix A of the linear system
        b (np.array): the vector result of Ax
        omega (int): overelaxation parameter
        precision (np.float, optional): The tolerance, below which the iterative scheme stops. Defaults to 10**-6.
        max_iter (int, optional): maximum number of iterations of the algorithm. Defaults to 2000.
        verbose (bool, optional): if true, print the results. Defaults to False.
        return_iter (bool, optional): return the solution x and the number of iterations. Defaults to False.

    Returns:
        Union[np.array, int]: the solutions of the linear system. If return_iter = True, it returns the number of 
        iteration as well.
    """

    start_time = time.time()
    n = len(A)
    x = np.random.uniform(size=n)
    iter_counter = 0
    error = np.finfo('double').max
    
    #calculate lower, upper and diagonal matrices
    L = - np.tril(A, -1)
    U = - np.triu(A, 1)
    D = np.diag(np.diag(A))
    
    #calculate M and N
    M = 1/omega * (D - omega*L)
    N = (1-omega)/omega * D + U
    
    #start iteration
    while error > precision and iter_counter <= max_iter:
        b_hat = N @ x + b
        x_next = np.zeros(n)
        #forward substitution
        for i in range(n):
            x_next[i] = (b_hat[i] - M[i, :i] @ x_next[:i]) / M[i, i]
        error = norm(x_next - x) / norm(x)
        x = x_next.copy()
        iter_counter += 1
    
    end_time = time.time()
    
    if error > precision:
        logging.warn(f'Iterative algorithm terminated due to max iter. Error = {error}')

    if verbose:
        print(f'# found solutions {x}')
        print(f'# in {iter_counter-1} iterations in {end_time - start_time} seconds')
        print(f'# absolute relative error {error}')
    
    if return_iter:
        return x, iter_counter  -1
    
    return x