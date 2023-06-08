import numpy as np
from math import sqrt
from QR_al import *
import matplotlib.pyplot as plt
def plot(index):
    plt.semilogy(abs(e1-ref)[:,index],label='no shift eigenvalues '+str(index+1))
    plt.semilogy(abs(e2-ref)[:,index],label='with shift eigenvalues '+str(index+1))
    plt.semilogy(abs(e3-ref)[:,index],label='with smart shift eigenvalues '+str(index+1))
    plt.legend()
    plt.show()
n=15
A = np.ones((n,n))
for i in range(n):
    A[n-i-1,n-i-1]+=i+1
_,e1 = QR(A,100,0)
_,e2 = QR(A,100,2)
_,e3 = QRshift(A,100)
ref = (np.sort(np.linalg.eigvals(A))[::-1])
plot(7)    

for i in range(7,14):
    plt.semilogy(abs(e3-ref)[:,i],label='eigenvalues '+str(i+1))
plt.legend()
plt.show()