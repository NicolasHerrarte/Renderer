import numpy as np
from Matrix import *

vectors = np.array([
[0],
[100],
[1000]
])

print(np.transpose([vectors[:,-1]]))
euler_rot = [0,0,2*math.pi/5]
quat = eulerToQuaternion(euler_rot)
rot_mat = rotQuatMatrix(quat)
for i in range(5):
    new_vector = np.matmul(rot_mat, np.transpose([vectors[:,-1]]))
    vectors = np.concatenate([vectors,new_vector], axis=1)
    print(vectors)

#print(2*math.pi/5)