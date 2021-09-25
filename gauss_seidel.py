import numpy as np
from numpy.core.defchararray import array
from scipy.linalg import norm, solve, block_diag
import time
from typing import *
import logging

def point_gauss_seidel(A: np.array, b: np.array, precision: np.float = 10**-6, max_iter: int = 2000, 
    verbose: bool = False, return_iter: bool  = False) -> Union[np.array, int]:
    """Implements the gauss seidel iterative algorithm and solves for Ax=b.
    Matrix A is broken down to two matrices. D is contains the diagonal elements and LU
    the non diagonal elements, such as A = D-L-U. The system that we solve transforms to
    (D-L)x^{k+1} = Ux^{k} + b where k is the the iteration step.

    Args:
        A (np.array): the matrix A of the linear system
        b (np.array): the vector result of Ax
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
    
    #extract the DL and U tables
    DL = np.tril(A)
    U = - np.triu(A, 1)

    while error > precision and iter_counter <= max_iter:
        #calculate the \hat{b}
        b_hat = U @ x + b
        #initialize the vector x^{k+1}
        x_next = np.zeros(n)
        #start forward substitution
        for i in range(n):
            x_next[i] = (b_hat[i] - DL[i, :i] @ x_next[:i]) / DL[i, i]
        #calculate the error
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
        return x, iter_counter-1
    return x

def block_gauss_seidel(A: np.array, b: np.array, block_size: int = 2, precision: np.float = 10**-6, max_iter: int = 2000, 
    verbose: bool = False, return_iter: bool  = False) -> Union[np.array, int]:
    """partitions the matrix A into square blocks as well as the vector b into blocks of the same size and 
    solves Ax = b. Matrix A is broken down to two matrices. D contains the diagonal blocks, whereas L and U
    are upper and lower block triangular matrices.

    Args:
        A (np.array): the matrix A of the linear system
        b (np.array): the vector result of Ax
        block_size (int): the size of the block
        precision (np.float, optional): The tolerance, below which the iterative scheme stops. Defaults to 10**-6.
        max_iter (int, optional): maximum number of iterations of the algorithm. Defaults to 2000.
        verbose (bool, optional): if true, print the results. Defaults to False.
        return_iter (bool, optional): return the solution x and the number of iterations. Defaults to False.

    Returns:
        Union[np.array, int]: the solutions of the linear system. If return_iter = True, it returns the number of 
        iteration as well.
    """
    start_time = time.time()
    iter_counter = 0
    error = np.finfo('double').max
    p = A.shape[0] // block_size
    n = len(A)
    
    #construct the upper triangular matrix
    U = -A.copy()
    for i in range(0,n,block_size):
        U[i:i+block_size, :i+block_size] = 0
    x = np.random.uniform(size=n)
    x_next = np.zeros(n)
    
    #start iteration
    while error > precision and iter_counter <= max_iter:
        #calculate the right-hand side
        b_hat = U @ x + b
        x_next = np.zeros(n)
        
        #start forward substitution
        x_next[0:block_size] = solve(A[:block_size, :block_size], b_hat[:block_size])
        for i in range(1, p):
            block_range = slice(i * block_size, (i + 1) * block_size)
            #get the diagonal block
            diag_block = A[block_range, block_range].copy()
            #calculate b tilted 
            b_tilted = b_hat[block_range] - A[block_range, :i * block_size] @ x_next[:i * block_size]
            #elimination for the values of x. the solve methods from scipy is used.
            x_next[block_range] = solve(diag_block, b_tilted)

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
        return x, iter_counter-1
    return x