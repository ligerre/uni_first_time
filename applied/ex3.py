from math import sqrt
import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg
import copy
def norm(z):
    return sqrt(z.dot(z))
def power_method(A,x0,ep=10**(-6),max_iter = 200):
    q = x0 / norm(x0)
    q2 = q
    err = []
    nu1 = []
    x1 = []
    res = ep+1
    niter = 0
    z = A@q
    while (res>=ep) and (niter<max_iter):
        q = z/ norm(z)
        z = A@q
        lam = q.dot(z)
        x1.append(q)
        z2 = q2@A
        q2 = z2 / norm(z2)
        q2 = np.transpose(q2)
        y1 = q2
        costheta = abs(y1.dot(q))
        if costheta >= (5*(10**(-2))):
            niter +=1
            res = norm(z-lam*q)/ costheta
            err.append(res)
            nu1.append(lam)
        else:
            print('Multiple eigenvalue')
            break
    return nu1, x1, err, niter

def basicqr(A,niter,tol=10**(-10)):
    T=A
    err = [np.diagonal(A)[:5]]
    for i in range(niter):
        Q,R = linalg.qr(T)
        T= R@Q
        val = np.diagonal(T)[:5]
        if norm(val-err[-1])>tol:
            err.append(val)
        else:
            break
    return T, err[1:]

n=100
A = np.random.uniform(0.0,1.0,size=(n,n))
x0 = np.random.random(n)

nu1, x1, err, niter = power_method(A,x0,ep = 10**(-12))

plt.semilogy(range(len(err)),err,label='error')
x_fin = x1[-1]
x1 = x1 - x_fin
print(nu1[-1])
nx1 = []
for i in range(len(x1)):
    nx1.append(norm(x1[i]))
plt.semilogy(range(len(nx1)),nx1,label='distance')
plt.legend()
plt.show()

A_symm = (A+A.T)/2


T, dia = basicqr(A_symm,5000)
ref = np.diagonal(T)[:5]
print(dia)
for i in range(5):
    plt.semilogy(range(len(dia)),abs(dia-ref)[:,i],label='eigenvalues'+str(i+1))
plt.legend()
plt.show()

plt.spy(T,precision=10**(-10))
plt.show()