import numpy as np
from numpy.core.defchararray import array
from scipy.linalg import norm, solve, block_diag
import time
from typing import *
import logging

def point_jacobi(
    A: np.array, b: np.array, precision: np.float = 10**-6, max_iter: int = 2000, 
    verbose: bool = False, return_iter: bool  = False) -> Union[np.array, int]:
    """Iplements the stanard point jacobi iterative algorithm, solving for Ax=b.
    Matrix A is broken down to two matrices. D is contains the diagonal elements and LU
    the non diagonal elements, such as A = D-L-U. The system that we solve transforms to
    Dx^{k+1} = (L+U)x^{k} + b where k is the the iteration step.
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
    x = np.random.uniform(size=len(A))
    iter_counter = 0
    
    # break the table as A = D - (LU)
    # D is the diagonal and LU = - (L + U)
    D = np.diag(A)
    LU = np.diagflat(D) - A
    
    error = np.finfo('double').max
    while error > precision and iter_counter <= max_iter:
        x_next = (b + (LU @ x)) / D
        iter_counter += 1
        error = norm(x_next - x) / norm(x)
        x = x_next
    
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

def block_jacobi( A: np.array, b: np.array, block_size: int = 2, precision: np.float = 10**-6, max_iter: int = 2000, 
    verbose: bool = False, return_iter: bool  = False) -> Union[np.array, int]:
    """partitions the matrix A into square blocks as well as the vector b into blocks of the same size and 
    solves Ax = b. Matrix A is broken down to two matrices. D contains the diagonal blocks, whereas L and U
    are upper and lower block triangular matrices. The algorithm solves simultaneously by performing gaussian
    elimination in parallel for each system/block.

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
    #dimension of the block table
    p = A.shape[0] // block_size
    n = len(A)
    
    #Calculate the block diagonal table
    A_blocked = A.reshape(p, block_size, p, block_size).swapaxes(1, 2)
    D_blocked = np.diagonal(A_blocked, axis1=0, axis2=1).T
    
    #calculate the LU table as Block Diagonal of A - A
    LU = block_diag(*D_blocked) - A
    
    #initialize the vector x
    x = np.random.uniform(size=n)
    
    #start the block jacobi iteration
    while error > precision and iter_counter <= max_iter:
        #initialize x^{k+1}
        x_next = np.zeros(shape=(p, block_size))
        
        #A hat is the block diagonal shape (p, block size, block size)
        A_hat = D_blocked.copy()
        
        #calculate the bhat and reshape to (p, block size)
        b_hat = LU @ x + b
        b_hat = b_hat.reshape(p, block_size)

        #start parallel gaussian elimination
        for i in range(block_size-1):
            #access all the  pivot elements
            pivot = A_hat[:, i, i]
            
            #eliminate all rows below the pivot
            for j in range(i+1, block_size):
                factor = A_hat[:, j, i] / pivot
                A_hat[:, j, :] -= np.multiply(A_hat[:, i, :].T, factor).T
                b_hat[:, j] -= b_hat[:, i] * factor
        
        #start backwards substitution
        x_next[:, block_size-1] = b_hat[:, block_size-1] / A_hat[:,block_size-1, block_size-1] #element wise division
        for k in range(block_size-2, -1, -1):
            #np.einsum('ij,ij->i',...,...) dot product within the blocks
            x_next[:, k] = (b_hat[:, k] - np.einsum('ij,ij->i', A_hat[:, k, k+1:], x_next[:, k+1:])) \
                           / A_hat[:, k, k] #element wise division
        
        #reshape x_next to original
        x_next = x_next.reshape(n)
        #calculate the error
        error = norm(x_next - x) / norm(x)
        x = x_next.copy()
        iter_counter += 1

    if error > precision:
        logging.warn(f'Iterative algorithm terminated due to max iter. Error = {error}')
    end_time = time.time()
    if verbose:
        print(f'# found solutions {x}')
        print(f'# in {iter_counter - 1} iterations in {end_time - start_time} seconds')
        print(f'# absolute relative error {error}')
    
    if return_iter:
        return x, iter_counter-1
    return x