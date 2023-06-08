import numpy as np
from math import sqrt
def vhouse(x):
    x = x/np.linalg.norm(x)
    s = x[1:].dot(x[1:])
    v = x
    if (abs(s)<=10^(-12)):
        beta =0.0
    else:
        #mu = sqrt(x[0]**2+s)
        mu=1
        if (x[0]<=0):
            v[0]=x[0]-mu
        else:
            v[0] = -s/(x[0]+mu)
        beta = 2/(s+v[0]**2)
       # v = v/v[0]
    return v,beta
def householder_reflection(A):
    n = A.shape[0]
    H = np.eye(n)
    for k in range(n-2):
        x = A[k+1:, k]
        v,beta = vhouse(x)
        """
        A[k+1:, k:] -= beta * np.outer(v, np.dot(v, A[k+1:, k:]))
        A[:, k+1:] -= beta * np.outer(np.dot(A[:, k+1:], v), v)
        H[k+1:, :] -= beta * np.outer(v, np.dot(v, H[k+1:, :]))
        """
        m = len(v)
        v = v.reshape(m,1)
        R = np.eye(m) - beta*(v@v.T)
        A[k+1:, k:] = R@A[k+1:, k:] 
        A[k+1:, k+1:] = A[k+1:, k+1:]@R
        H[k+1:, :] = R@H[k+1:, :]
    main_diag = np.diagonal(A).copy()
    off_diag = np.diagonal(A,offset=-1).copy()
    A = np.diag(main_diag)+np.diag(off_diag,-1)+np.diag(off_diag,1)
    return A, H

# Example usage
A = np.array([[4, 3, 2, 1],
              [3, 4, 3, 2],
              [2, 3, 4, 3],
              [1, 2, 3, 4]]).astype('float64')
print("Original matrix:\n", A)
B = A.astype('float64')
H, Q = householder_reflection(B)
print("Upper Hessenberg matrix:\n", H)
print("Different \n", (Q.T@H@Q - A).round(9))

v = A[1:,1]
u = np.outer(v,v)
v= v.reshape(len(v),1)
print((v@v.T - u))