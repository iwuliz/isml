from cvxopt import matrix
from cvxopt import spmatrix
from cvxopt import sparse
from cvxopt import spdiag
import numpy as np


A = matrix([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], (2,3))
print(A)
A[::5] *= -1



'''
B = matrix([ [1.0, 2.0], [3.0, 4.0] ])
print(B)

C = matrix([ [A] ,[B] ])
print(C)

D = spmatrix([1.,2., 3.], [0, 1, 2], [0, 1, 0], (4,2))
print(D)
print(matrix(D))

E = sparse([ [B, B], [D] ])
print(E)

F = spdiag([B, -B, 1, 2, 3, 4])
print(F)
'''

F = spdiag([1, 2, 3, 4])
print(F)

'''
G = matrix(range(16), (4, 4))
print(G)


Gi = G[0,:]*0
for i in range(1, 4):
    print(G[i,:])
    Gtmp = G[i,:] * i
    print(Gtmp)
    Gi = matrix([[Gi, Gtmp]])
print(Gi)

Gij = Gi[:,0]*0
for j in range(1,4):
    print(Gi[:,j])
    Gtmp = Gi[:,j]*j
    print(Gtmp)
    Gij = matrix([[Gij],[Gtmp]])
print(Gij)


G[::5] *= -1
print(G)

H = G.trans()
print(H)

'''