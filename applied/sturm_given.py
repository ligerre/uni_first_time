import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
def sturm(d,b,x):
    n = len(d)
    p = np.zeros(n+1)
    p[0]=1
    p[1]= d[0]-x
    for i in range(1,n):
        p[i+1] = (d[i]-x)*p[i] - b[i-1]*b[i-1]*p[i-1]
    return p
def checksign(dd,bb,x):
    p= sturm(dd,bb,x)
    n = len(dd)
    nch=0
    s=0
    for i in range(n):
        if p[i+1]*p[i]<=0:
            nch+=1
        if p[i+1]==0:
            s+=1
    return nch-s
def bound(dd,bb):
    n = len(dd)
    alpha = dd[0] - abs(bb[0])
    temp = dd[n-1] - abs(bb[n-2])
    if temp < alpha:
        alpha = temp
    for i in range(1,n-1):
        temp = dd[i]-abs(bb[i-1])-abs(bb[i])
        if temp < alpha:
            alpha = temp
    
    beta = dd[0] + abs(bb[0])
    temp = dd[n-1] + abs(bb[n-2])
    if temp >beta:
        beta= temp
    for i in range(1,n-1):
        temp = dd[i]+abs(bb[i-1])+abs(bb[i])
        if temp >beta:
            beta= temp
    
    return alpha, beta
def given_sturm(dd,bb,ind,toll):
    a,b = bound(dd,bb)
    dist = abs(b-a)
    s = abs(a)+abs(b)
    n = len(dd)
    niter = 0
    nch = []
    ak = []
    bk = []
    ck = []
    while dist > toll*s:
        niter += 1
        c = (b+a)/2
        ak.append(a)
        bk.append(b)
        ck.append(c)
        nch.append(checksign(dd,bb,c))
        if nch[-1]>n-ind:
            b=c
        else:
            a=c

        dist = abs(b-a)
        s = abs(a)+abs(b)
    return ak,bk,ck,nch,niter

toll = 10**(-15)
n = 20
ind = 7
main_diag = np.concatenate([np.ones(1),2*np.ones(n-2),np.ones(1)]) 
off_diag = -np.ones(n-1)
#print(main_diag)
T = np.diag(main_diag) + np.diag(off_diag,-1)+ np.diag(off_diag,1)
#print(T)
ak,bk,ck,nch,niter = given_sturm(main_diag,off_diag,ind,toll)

eigval = (np.sort(np.linalg.eigvals(T))[::-1])[ind-1]*np.ones(len(ak))
plt.semilogx(ak,label='a')
plt.semilogx(bk,label='b')
plt.semilogx(eigval,label='true_val')
plt.legend()
plt.show()

aa = np.array([4.0,3.0,2.0,1.0])
bb = np.ones(3)
a,b,c,_,_=given_sturm(aa,bb,3,toll)
print(c[-1])
cc = []
for i in range(20):
    _,_,ck,_,_ = given_sturm(main_diag,off_diag,i+1,toll)
    cc.append(ck[-1])
print(cc)
v = np.sort(np.linalg.eigvals(T))[::-1]
for i in range(len(v)):
    v[i]-= cc[i]
print(v)
