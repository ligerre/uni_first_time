import numpy as np
from math import sqrt
def vhouse(x):
    if np.linalg.norm(x[1:])==0:
        return x
    else:
        v = np.zeros_like(x)
        v[0] = np.copysign(np.linalg.norm(x), -x[0])
        v = v + x
        return v
def householder_reflection(A):
    m, n = A.shape
    H = np.eye(m)
    for k in range(n-2):
        x = A[k+1:, k]
        v = vhouse(x)
        v = v / np.linalg.norm(v)
        A[k+1:, k:] -= 2 * np.outer(v, np.dot(v, A[k+1:, k:]))
        A[:, k+1:] -= 2 * np.outer(np.dot(A[:, k+1:], v), v)
        H[k+1:, :] -= 2 * np.outer(v, np.dot(v, H[k+1:, :]))
    return A, H
def givcos(x,y):
    if y==0:
        return 1,0
    else:
        if abs(y)>abs(x):
            t = -x/y
            s = 1/sqrt(1+t*t)
            c = s*t
        else:
            t = -y/x
            c = 1/sqrt(1+t*t)
            s = c*t
    return c,s
def qr_hessenberg(H):
    n = H.shape[0]
    Q = np.identity(n)
    R = np.copy(H)
    for j in range(n-1):
        i=j+1
        if R[i,j] != 0:
            x, y = R[j,j], R[i,j]
            #c = x / np.sqrt(x**2 + y**2)
           # s = y / np.sqrt(x**2 + y**2)
            c,s = givcos(x,y)
            G = np.array([[c, -s], [s, c]])
            R[j:j+2,:] = G @ R[j:j+2,:]
            Q[:,j:j+2] = Q[:,j:j+2] @ G.T
    return Q, R
def QR(A,niter,mu):
    B = np.copy(A)
    T,_ = householder_reflection(B)
    shift = mu*np.eye(A.shape[0])
    for i in range(niter):
        Q,R = qr_hessenberg(T-shift)
        T = R@Q+shift
        if niter <=10:
            print('iteration: '+str(i+1),'\n',(np.sort(np.diag(T))[::-1])[:5])
    return T