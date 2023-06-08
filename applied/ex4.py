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

def hessenberg_symmetric(A):
    H = np.copy(A)
    H = H.astype('float64')
    Q = np.eye(A.shape[0])
    for k in range(H.shape[0]-2):
        x = H[k+1:, k]
        v = vhouse(x)
        v = v / np.linalg.norm(v)
        H[k+1:, k:] -= 2 * np.outer(v, np.dot(v, H[k+1:, k:]))
        H[:, k+1:] -= 2 * np.outer(np.dot(H[:, k+1:], v), v)
    return H,Q
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
def QR(A,niter):
    T,_ = householder_reflection(A)
    for i in range(niter):
        Q,R = qr_hessenberg(T)
        T = R@Q
        if niter <=10:
            print('iteration: '+str(i+1),'\n',T)
    return T

A = np.array([[4, 3, 2, 1],
              [3, 4, 3, 2],
              [2, 3, 4, 3],
              [1, 2, 3, 4]]).astype('float64')
A = np.array([[13,4,3,9],[-1,-8,5,0],[2, 3, 8,1],[6,-2,0,4]]).astype('float64')
#A = np.array([[2, 1, 0, 0],[1, 2, 1, 0],[0, 1, 2, 1],[0, 0, 1, 2]]).astype('float64')
print("Original matrix:\n", A)
B = A.astype('float64')
H, Q = householder_reflection(B)
print("Upper Hessenberg matrix:\n", H)
print("Different \n", (Q.T@H@Q - A).round(9))
#B = A.astype('float64')
#B = np.array([[0,1],[-1,0]]).astype('float64')
B=np.array([[13, 4, 3, 9],[-1, -8, 5, 0],[2, 3, 8, 1],[6, -2, 0, 4]]).astype('float64')
print("eigenvalues: \n", np.linalg.eigvals(B))
T = QR(B,10)
#print("After 10 iteration \n", T)
