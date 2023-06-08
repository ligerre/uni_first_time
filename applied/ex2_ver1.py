from time import time
from math import sqrt
from sksparse.cholmod import cholesky
from scipy.sparse import diags, kron, eye, dia_matrix, csc_matrix
from scipy.sparse.linalg import cg, spilu, LinearOperator
import numpy as np
import sys
import matplotlib.pyplot as plt


def timeit(func: callable, args: list):
    start = time()
    result = func(*args)
    end = time()
    print("\nTime:", end - start)
    return result

def residual(xk):
    error_cg.append(sqrt((xk-x_star).dot(A.dot(xk-x_star))))

n = 10**2
N = n*n

diag = 2*np.ones(n)
subdiag = -np.ones(n-1)

I = eye(n)
K_d1 = diags([subdiag, diag, subdiag], offsets=[-1, 0, 1], format="csc")
A = kron(I, K_d1) + kron(K_d1, I)
A = csc_matrix(A)
x_0 = np.zeros(N)
x_star = np.random.random(N)
b = A.dot(x_star)

epsilon = 10**(-6)
max_iter = 2500
error_cg=[sqrt((x_0-x_star).dot(A.dot(x_0-x_star)))]
x_fin = cg(A,b,x0=x_0,callback=residual)

e_0 = error_cg[0]
error_cg = np.array(error_cg)
plt.semilogy(range(len(error_cg)), error_cg/e_0, 'r', label="CG")

error_cg=[sqrt((x_0-x_star).dot(A.dot(x_0-x_star)))]
ilu = spilu(A)
Mx = lambda x: ilu.solve(x)
M = cholesky(A)

x_fin2 = cg(A,b,x0=x_0, M=M(), maxiter=max_iter, callback=residual)

e_0 = error_cg[0]
error_cg= np.array(error_cg)
plt.semilogy(range(len(error_cg)), error_cg/e_0, 'b', label="PCG")
plt.legend()
plt.savefig("./plot1.png")
plt.close()
