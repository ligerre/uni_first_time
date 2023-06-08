import numpy as np    
import matplotlib.pyplot as plt

class LinearAdvection1D:
    # Matrix for LA1D 
   A=0
   # Initialization of constants 
   def __init__(self, c, x0, xN, N, deltaT,T):
      self.c = c 
      self.x0 = x0   
      self.xN = xN 
      self.N = N   
      self.deltaT = deltaT   
      self.T = T       
   # CFL number funct.   
   def CFL(self):
       deltaX= (self.xN - self.x0)/self.N
       return self.c*self.deltaT/deltaX
   # check CFL number <=1 or not.
   def checkCFL(self):
       if (np.abs(self.CFL())<=1):
           flag=True 
       else:
           flag=False
       return flag
   # Matrix assembly of LA1D   
   def upwindMatrixAssembly(self):
       alpha_min=min(self.CFL(),0)
       alpha_max=max(self.CFL(),0)
       a1=[alpha_max]*(self.N-1)
       a2=[1+alpha_min-alpha_max]*(self.N)
       a3=[-alpha_min]*(self.N-1)
       self.A=np.diag(a1, -1)+np.diag(a2, 0)+np.diag(a3, 1)
       self.A[0,-1]=alpha_max
       self.A[N-1,0]=-alpha_min
   # Solve u=Au0
   def Solve(self,u0):
       return np.matmul(self.A,u0) 
class Lax_Friedrich:
    # Matrix for LA1D 
   A=0
   # Initialization of constants 
   def __init__(self, c, x0, xN, N, deltaT,T):
      self.c = c 
      self.x0 = x0   
      self.xN = xN 
      self.N = N   
      self.deltaT = deltaT   
      self.T = T       
   # CFL number funct.   
   def CFL(self):
       deltaX= (self.xN - self.x0)/self.N
       return self.c*self.deltaT/deltaX
   # check CFL number <=1 or not.
   def checkCFL(self):
       if (np.abs(self.CFL())<=1):
           flag=True 
       else:
           flag=False
       return flag
   # Matrix assembly of LA1D   
   def MatrixAssembly(self):
       alpha = self.CFL()
       a1=[(1-alpha)/2]*(self.N-1)
       a3=[(1+alpha)/2]*(self.N-1)
       self.A=np.diag(a1, 1)+np.diag(a3, -1)
       self.A[0,-1]=(1+alpha)/2
       self.A[N-1,0]=(1-alpha)/2
   # Solve u=Au0
   def Solve(self,u0):
       return np.matmul(self.A,u0) 
#############  
# Start of the code
###################

# constants  
N,x0,xN,deltaT,c,T=4000,0.,1.,0.0002,-1.,0.5
# initialization of constants
LA1D = LinearAdvection1D(c, x0, xN, N, deltaT,T) 

# initial value
x=np.linspace(LA1D.x0,LA1D.xN,LA1D.N)
u0=lambda x: np.sin(2*np.pi*x)
u_ref = u0(x)
u_t = lambda t: u0(x+t)
#plot of initial value    
plt.plot(x,u0(x),label="Initial value")
plt.ylabel('u')
plt.xlabel('x')
plt.legend()


# calculating solution if CFL<=1
if (LA1D.checkCFL() is True):
    print("CFL number is: ", LA1D.CFL())
    LA1D.upwindMatrixAssembly()
    for t in range(0,int(LA1D.T/LA1D.deltaT)):
        u=LA1D.Solve(u_ref)
        u_ref=u
else:
    print("CFL number is greater than 1. CFL: ", LA1D.CFL())

# ploting the last solution
plt.plot(x,u,label="Solution at t="+str(LA1D.T))
plt.legend()
plt.grid(linestyle='dotted')
#########################################
TD = Lax_Friedrich(c, x0, xN, N, deltaT,T)
u_ref = u0(x)
if (TD.checkCFL() is True):
    print("CFL number is: ", LA1D.CFL())
    TD.MatrixAssembly()
    for t in range(0,int(TD.T/TD.deltaT)):
        u1=TD.Solve(u_ref)
        u_ref=u1
else:
    print("CFL number is greater than 1. CFL: ", TD.CFL())
###################################
plt.plot(x,u1,label="Solution Lax Friedrich at t="+str(LA1D.T))
plt.legend()

plt.plot(x,u_t(LA1D.T),label="reference Solution at t="+str(LA1D.T))
plt.legend()
plt.savefig('LA1D.png',dpi=300)
plt.close()
plt.plot(x,np.abs(u_t(LA1D.T)-u),label="Error of Upwind")
plt.legend()
plt.grid(linestyle='dotted')
plt.plot(x,np.abs(u_t(LA1D.T)-u1),label="Error of Lax_Friedrich")
plt.legend()
plt.savefig("error_1d.png",dpi=300)