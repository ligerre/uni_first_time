import numpy as np
from math import sqrt
import matplotlib.pyplot as plt

def eval(f,g,x,d,scalar = True):
    sigma = 0.3
    beta = 0.5
    t = beta ** (0)
    if scalar:
        while np.linalg.norm(f(x+t*d))>np.linalg.norm(f(x)+sigma*t*g*d):
            t = t*beta
    else:
        while np.linalg.norm(f(x+t*d))>np.linalg.norm(f(x)+sigma*t*g@d):
            t = t*beta
    return x+t*d
def newton(f,g,x,scalar=False):
    iter = 0
    x_list = [x]
    while iter < nmax:
        
        f_current = f(x)
        #print(f_current)
        if np.linalg.norm(f_current)<ep:
            break
        grad = g(x)
        
        if scalar:
            d = -f_current/grad
            if np.linalg.norm(d)<ep:
                break
            x = eval(f,grad,x,d)
        else:
            d = -np.linalg.inv(grad)@f_current
            if np.linalg.norm(d)<ep:
                break
            #x+=d
            x = eval(f,grad,x,d,False)
        iter+=1
        x_list.append(x.copy())
    return x_list,iter
def broyden(f,Q,x,scalar = False):
    iter = 0
    f_current = f(x)
    x_list = [x]
    while iter<nmax:
        if scalar:
            delta = -f_current/Q
            if abs(delta)<np.sqrt(ep):
              break
            x = eval(f,Q,x,delta,scalar)
            f_current = f(x)
            Q = Q + f_current*delta/(delta*delta)
        else:
            delta = -np.linalg.solve(Q,f_current)
            if np.linalg.norm(delta)<np.sqrt(ep):
              break
            #x += delta
            x = eval(f,Q,x,delta,scalar)
            f_current = f(x)
            #f_new = f(x)
            #b = f_new - f_current
            #Q = Q + np.outer(b-Q@delta,delta)/np.dot(delta,delta)
            Q = Q + np.outer(f_current,delta)/np.dot(delta,delta)
            #f_current = f_new
        iter+=1
        x_list.append(x.copy())
    return x_list,iter
def good_broyden(f,Q,x,scalar = False):
    iter = 0
    f_current = f(x)
    x_list = [x]
    while iter<nmax:
        if scalar:
            delta = -f_current/Q
            if abs(delta)<np.sqrt(ep):
              break
            x = eval(f,Q,x,delta,scalar)
            f_current = f(x)
            Q = Q + f_current*delta/(delta*delta)
        else:
            delta = -Q@f_current
            if np.linalg.norm(delta)<np.sqrt(ep):
              break
            x += delta
            #x = eval(f,Q,x,delta,scalar)
            f_new = f(x)
            b = f_new - f_current
            Q = Q + np.outer(delta - Q@b,delta.T@Q)/np.dot(delta.T,Q@b)
            f_current = f_new
        iter+=1
        x_list.append(x.copy())
    return x_list,iter
nmax = 200
ep = 10**(-16)

f = lambda x: np.array([x[0]-x[1],x[0]*x[1]])
g = lambda x: np.array([[1,-1],[x[0],x[1]]])
#print(g([1,1]))
x_list1,iter1 = newton(f,g,[1,1])
Q = g([1,1])
x_list2,iter = broyden(f,Q,[1,1])
x_list3,iters = good_broyden(f,np.linalg.inv(Q),[1,1])
print(iter)
print(x_list2)
print(iters)
print(x_list3)
xx = np.linspace(-0.2,1.2,50)
yy = np.linspace(-0.2,1.2,50)
X,Y = np.meshgrid(xx,yy)
Z = np.linalg.norm(f([X,Y]),axis = 0)

plt.contourf(X,Y,Z)
x_cord = [x[0] for x in x_list2]
y_cord = [x[1] for x in x_list2]
plt.scatter(x_cord,y_cord)
plt.show()

f1 = lambda x: np.arctan(x)
g1 = lambda x: 1/(1+x*x)
x_list,iter2 = broyden(f1,g1(2),2,True)
print(iter2)
print(x_list)
print(np.arctan(x_list))
plt.plot(x_list)
plt.show()
