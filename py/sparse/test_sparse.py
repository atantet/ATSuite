import numpy as np
from scipy import linalg, sparse
import time

N = 100
rho = 0.1

tau = 0.99

E = int(N**2 * rho)

print time.strftime("%H:%M:%S")
Ra = np.sort(np.random.randint(0, high=N, size=E))
Ca = np.random.randint(0, high=N, size=E)
Va = np.random.rand(E)
asp = sparse.csr_matrix((Va, np.array([Ra, Ca])), shape=(N, N))
a = asp.todense()

print time.strftime("%H:%M:%S")
Rb = np.sort(np.random.randint(0, high=N, size=E))
Cb = np.sort(np.random.randint(0, high=N, size=E))
Vb = np.random.rand(E)
bsp = sparse.csr_matrix((Vb, np.array([Rb, Cb])), shape=(N, N))
b = bsp.todense()

#b = np.random.rand(N, N)
#b[b < tau] = 0.

# print time.strftime("%H:%M:%S")
# c = linalg.blas.dgemm(1., a.T, b.T, trans_a=True, trans_b=True) 
# print time.strftime("%H:%M:%S")

#asp = sparse.csr_matrix(a)
#bsp = sparse.csr_matrix(b)

print time.strftime("%H:%M:%S")
csp = asp * bsp
print time.strftime("%H:%M:%S")

c = a * b

print np.sum((csp.todense() - c)**2)
