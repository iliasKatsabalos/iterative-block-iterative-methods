
from gauss_seidel import block_gauss_seidel, point_gauss_seidel
from re import A
import numpy as np
from jacobi import point_jacobi, block_jacobi
from gauss_seidel import point_gauss_seidel
from sor import SOR

n = 32
T = np.zeros((n**2, n**2))

diag_ind = np.diag_indices(n**2)
T[diag_ind] = 4
lo_diag_ind_rows, lo_diag_ind_columns = np.diag_indices(n**2 - 1)
lo_diag_ind_rows = lo_diag_ind_rows + 1
T[lo_diag_ind_rows, lo_diag_ind_columns] = -1

up_diag_ind_rows, up_diag_ind_columns = np.diag_indices(n**2 - 1)
up_diag_ind_columns = up_diag_ind_columns + 1
T[up_diag_ind_rows, up_diag_ind_columns] = -1

#lower identity
for i in range(n, n**2, n):
    T[i:i+n, i-n:i] = -1 * np.identity(n)

#upper identity
for i in range(0, n**2 -n, n):
    T[i:i+n, i+n:i+2*n] = -1 * np.identity(n)

b = np.ones(n ** 2)


print ('Solving uisng point jacobi')
res = point_jacobi(A=T, b=b, precision=1e-4, max_iter=2000, verbose=True, return_iter=True)
print('============================')
print ('Solving uisng block jacobi')
res = block_jacobi(A=T, b=b, block_size=2, precision=1e-4, max_iter=5000, verbose=True, return_iter=True)
print('============================')
print ('Solving uisng point gauss-seidel')
res = point_gauss_seidel(A=T, b=b, precision=1e-4, max_iter=2000, verbose=True, return_iter=True)
print('============================')
print ('Solving uisng block gauss-seidel')
res = block_gauss_seidel(A=T, b=b, block_size=2, precision=1e-4, max_iter=5000, verbose=True, return_iter=True)
print('============================')
print ('Solving uisng sor')
res = SOR(A=T, b=b, omega=1.8, precision=1e-4, max_iter=5000, verbose=True, return_iter=True)
