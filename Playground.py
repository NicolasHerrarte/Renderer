import numpy as np
from Matrix import *
perm = permMatrix([1, -1], 3)
filtered = np.transpose(np.array([x for x in np.transpose(perm) if x[2]==-1]))
np.concatenate([filtered, np.array([[0],[0],[1]])],axis=1)
