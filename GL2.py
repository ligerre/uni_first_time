import numpy as np
def gen(A,k):
    base = np.eye(2)
    for i in range(k):
        base = base@A 
    return base, np.linalg.inv(base)
A = np.array([[2,0],[0,3]])
B = np.array([[3,5],[0,5]])

k1 = np.random.randint(10)
k2 = np.random.randint(10)


G,G1 = gen(A,k1+1)

F,F1 = gen(B,k1+1)
print(k1+1)
print(G@F@G1@F1)