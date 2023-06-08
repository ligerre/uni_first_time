from time import time
from math import sqrt

from scipy.sparse import diags, kron, eye, dia_matrix, csc_matrix, csr_matrix
from scipy.sparse.linalg import spsolve_triangular, spilu
from sksparse.cholmod import cholesky
import numpy as np
import sys
import matplotlib.pyplot as plt


def timeit(func: callable, args: list):
    start = time()
    result = func(*args)
    end = time()
    print("\nTime:", end - start)
    return result


def sparse_cholesky(A): # The input matrix A must be a sparse symmetric positive-definite.
  
  n = A.shape[0]
  LU = spilu(A) # sparse LU decomposition
  

  return LU.L.dot( diags(LU.U.diagonal()**0.5) )

def conjugate_gradient(A: dia_matrix,
                       b: np.ndarray,
                       x_0: np.ndarray,
                       epsilon: float,
                       max_iter: int,
                       x_star: np.ndarray):
    x = x_0.copy()

    error_seq = [sqrt((x_0 - x_star).dot(A.dot(x_0 - x_star)))]

    res = b - A.dot(x)
    p = res

    k = 0

    norm_squared = res.dot(res)

    while sqrt(norm_squared) > epsilon and k < max_iter:
        
        A_dot_p = A.dot(p)

        alpha = norm_squared/(p.dot(A_dot_p))

        x += alpha*p

        res -= alpha*A_dot_p

        old_norm_squared = norm_squared
        norm_squared = res.dot(res)

        beta = norm_squared/old_norm_squared

        p = res + beta*p

        error = x - x_star
        error_seq.append(sqrt(error.dot(A.dot(error))))

        if k % 10 == 0:
            print(k, sqrt(norm_squared),"   ", end="\r")

        k += 1

    return x, k, error_seq

def pre_con(b):
    c = spsolve_triangular(L,b)
    res = spsolve_triangular(L.transpose(),c,lower=False) 

    return res
def pre_conjugate_gradient(A: dia_matrix,
                       b: np.ndarray,
                       x_0: np.ndarray,
                       epsilon: float,
                       max_iter: int,
                       x_star: np.ndarray):
    x = x_0.copy()
    error_seq = [sqrt((x_0 - x_star).dot(A.dot(x_0 - x_star)))]

    res = b - A.dot(x)
    res_hat = M(res)
    p = res_hat

    k = 0

    norm_squared = res.dot(res_hat)

    while sqrt(norm_squared) > epsilon and k < max_iter:
        
        A_dot_p = A.dot(p)

        alpha = norm_squared /(p.dot(A_dot_p))

        x += alpha*p

        res -= alpha*A_dot_p
        res_hat = M(res)
        
        old_norm_squared = norm_squared
        norm_squared = res.dot(res_hat)

        beta = norm_squared/old_norm_squared

        p = res_hat + beta*p

        error = x - x_star
        error_seq.append(sqrt(error.dot(A.dot(error))))

        if k % 10 == 0:
            print(k, sqrt(norm_squared),"   ", end="\r")

        k += 1

    return x, k, error_seq

n = 10**3
N = n*n

diag = 2*np.ones(n)
subdiag = -np.ones(n-1)

I = eye(n)
K_d1 = diags([subdiag, diag, subdiag], offsets=[-1, 0, 1], format="csc")
A = kron(I, K_d1) + kron(K_d1, I)
A = csc_matrix(A)
M = cholesky(A)
x_0 = np.zeros(N)
x_star = np.random.random(N)
b = A.dot(x_star)
epsilon = 10**(-6)
max_iter = 2500

print("\nConjugate Gradient")
x, k, error_seq = timeit(conjugate_gradient, [
                         A, b, x_0, epsilon, max_iter, x_star])
e_0 = error_seq[0]
error_seq = np.array(error_seq)

print("iterations:", k, "\nerror norm:", np.linalg.norm(
    x - x_star), "\nerror energy norm:", error_seq[-1])
plt.semilogy(range(len(error_seq)), error_seq/e_0, 'r', label="CG")
x_0 = np.zeros(N)
b = A.dot(x_star)
print("\nPreCon Conjugate Gradient")
x, k, error_seq = timeit(pre_conjugate_gradient, [
                         A, b, x_0, epsilon, max_iter, x_star])
e_0 = error_seq[0]
error_seq = np.array(error_seq)
e = A.dot(x)-b
print("error",e.dot(e))
print("iterations:", k, "\nerror norm:", np.linalg.norm(
    x - x_star), "\nerror energy norm:", error_seq[-1])
plt.semilogy(range(len(error_seq)), error_seq/e_0, 'b', label="preCG")
plt.legend()
plt.savefig("./plot.png")
plt.close()
