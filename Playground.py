import numpy as np
np.set_printoptions(suppress=True)

A = np.array([[5,6],[1,3]])
AA = np.array([[4, 5,6],[2, 1,3]])
A_INV = np.linalg.inv(A)
print(np.matmul(A_INV, AA))

#