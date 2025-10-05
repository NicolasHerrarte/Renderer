import numpy as np
import math
from colorama import Fore, Back, Style

np.set_printoptions(suppress=True)

EQUALITY_NUM = 1.e-10
def permMatrix(nums, iterations):
    l = np.array(nums)
    n = l.shape[0]
    lower_gear = np.expand_dims(l, axis=0)

    for i in range(iterations-1):
        next_gear_lower = np.tile(lower_gear, n)
        shape = (1,n**(i+2))
        next_gear_upper = np.reshape(np.repeat(l, n**(i+1), axis=0), shape)
        next_gear = np.concatenate([next_gear_upper, next_gear_lower], axis=0)

        lower_gear = next_gear

    return lower_gear

def switchPermutation(size, ind1, ind2):
    identity = np.identity(size)
    identity[ind1, ind1] = 0
    identity[ind2, ind1] = 1
    identity[ind2, ind2] = 0
    identity[ind1, ind2] = 1
    return identity

def colShiftMatrix(size, shifts):
    I = np.identity(size)
    for i in range(shifts):
        tmp = np.copy([I[-1]])
        I = np.concatenate([tmp, I], axis=0)
        I = I[0:-1]
    return I

def overlapHalvesMatrix(M):
    MT = np.transpose(M)
    addM_MT = M+MT
    rows, cols = M.shape
    for i in range(rows):
        for j in range(cols):
            if (j > i):
                addM_MT[i, j] = 0
    return addM_MT

def rotEulerMatrix(euler, decimals=8):
    euler = np.round(euler, decimals)
    x, y, z = euler
    x = np.array([[1, 0, 0], [0, np.cos(np.radians(x)), -np.sin(np.radians(x))],
                [0, np.sin(np.radians(x)), np.cos(np.radians(x))]])
    y = np.array([[np.cos(np.radians(y)), 0, np.sin(np.radians(y))], [0, 1, 0],
                  [-np.sin(np.radians(y)), 0, np.cos(np.radians(y))]])
    z = np.array([[np.cos(np.radians(z)), -np.sin(np.radians(z)), 0], [np.sin(np.radians(z)), np.cos(np.radians(z)), 0],
                  [0, 0, 1]])

    x = np.round(x, decimals=decimals)
    y = np.round(y, decimals=decimals)
    z = np.round(z, decimals=decimals)

    return x, y, z

def eulerToQuaternion(euler, decimals=8):
    euler = np.round(euler, decimals)
    x, y, z = euler

    q0 = np.cos(x/2)*np.cos(y/2)*np.cos(z/2)+np.sin(x/2)*np.sin(y/2)*np.sin(z/2)
    q1 = np.sin(x/2)*np.cos(y/2)*np.cos(z/2)-np.cos(x/2)*np.sin(y/2)*np.sin(z/2)
    q2 = np.cos(x/2)*np.sin(y/2)*np.cos(z/2)+np.sin(x/2)*np.cos(y/2)*np.sin(z/2)
    q3 = np.cos(x/2)*np.cos(y/2)*np.sin(z/2)-np.sin(x/2)*np.sin(y/2)*np.cos(z/2)

    quat = np.array([q0,q1,q2,q3])
    quat = np.round(quat, decimals=decimals)

    return quat

def quaternionToEuler(quat, decimals=8):
    quat = np.round(quat, decimals=decimals)
    q0, q1, q2, q3 = quat

    y = np.arcsin(np.round(2 * (q0 * q2 - q1 * q3),5))
    if abs(y - (math.pi/2)) < EQUALITY_NUM:
        x = 0
        z = -2*np.arctan(q1/q0)
    elif abs(y - (-math.pi/2)) < EQUALITY_NUM:
        x = 0
        z = 2 * np.arctan(q1 / q0)
    else:
        x = np.arctan((2*(q0*q1+q2*q3))/(q0**2-q1**2-q2**2+q3**2))
        z = np.arctan((2*(q0*q3+q1*q2))/(q0**2+q1**2-q2**2-q3**2))

    euler = np.array([x, y, z])
    euler = np.round(euler, decimals=decimals)
    return euler

def axisToQuaternion(axis_rotation, decimals=8):
    axis_rotation = np.round(axis_rotation, decimals=decimals)
    theta, x, y, z = axis_rotation

    q0 = np.cos(theta/2)
    q1 = x*np.sin(theta/2)
    q2 = y*np.sin(theta/2)
    q3 = z*np.sin(theta/2)

    quat = np.array([q0, q1, q2, q3])
    quat = np.round(quat, decimals=decimals)
    return quat
def rotQuatMatrix(quat, decimals=8):
    quat = np.round(quat, decimals=decimals)
    q0, q1, q2, q3 = quat
    m = np.array([[q0**2+q1**2-q2**2-q3**2, 2*q1*q2-2*q0*q3,2*q1*q3+2*q0*q2],
         [2*q1*q2+2*q0*q3,q0**2-q1**2+q2**2-q3**2,2*q2*q3-2*q0*q1],
         [2*q1*q3-2*q0*q2,2*q2*q3+2*q0*q1,q0**2-q1**2-q2**2+q3**2]])

    m = np.round(m, decimals=decimals)
    return m

def quatMult(quat1, quat2, decimals=8):
    quat1 = np.round(quat1, decimals=decimals)
    quat2 = np.round(quat2, decimals=decimals)

    r0, r1, r2, r3 = quat1
    s0, s1, s2, s3 = quat2

    t0 = r0 * s0 - r1 * s1 - r2 * s2 - r3 * s3
    t1 = r0 * s1 + r1 * s0 + r2 * s3 - r3 * s2
    t2 = r0 * s2 - r1 * s3 + r2 * s0 + r3 * s1
    t3 = r0 * s3 + r1 * s2 - r2 * s1 + r3 * s0

    t = np.array([t0, t1, t2, t3])
    t = np.round(t, decimals=decimals)
    return t

def quatInv(quat, decimals=8):
    quat = np.round(quat, decimals=decimals)
    q0, q1, q2, q3 = quat

    return np.array([q0, -q1, -q2, -q3])

def focalMatrix(focus):
    focus = np.array([[-focus, 0, 0],[0,-focus,0], [0,0,1]])
    return focus

def displayVectors(vectors, special=False):
    vectors = np.transpose(vectors)
    for v in vectors:
        v = [str(x) for x in v]
        if special:
            print(Fore.BLUE+"("+",".join(v)+")"+Style.RESET_ALL)
        else:
            print("(" + ",".join(v) + ")")

def binMatrixToVectors(matrix):
    n, _ = matrix.shape
    vectors = []
    for i in range(n):
        for j in range(n):
            if matrix[i, j] == 1:
                vectors.append([j, i])

    return np.transpose(np.array(vectors))

def extendVectorbyMagnitud(vector, magnitud):
    m = np.linalg.norm(vector)
    ratio = (m+magnitud)/m
    new_vector = vector*ratio
    return np.array(new_vector)

def getBase3Points(vectors, mode):
    a = vectors[:, 0]
    b = vectors[:, 1]
    c = vectors[:, 2]

    u = np.expand_dims(a - b, axis=0)
    v = np.expand_dims(a - c, axis=0)
    A = np.transpose(np.concatenate([u, v], axis=0))

    if mode == "REGULAR":
        return A
    elif mode == "ORTHONORMAL":
        Q, _ = np.linalg.qr(A)
        return Q

def getNormalBase(vectors, unit=True):
    u = vectors[:, 0]
    v = vectors[:, 1]
    cross_vector = np.cross(v, u)
    if not unit:
        return cross_vector
    else:
        normal_vector = cross_vector / np.linalg.norm(cross_vector)
        return normal_vector

def projectionMatrixBase(base, mode):
    if mode == "REGULAR":
        A = base
        AT = np.transpose(A)
        inv_ATA = np.linalg.inv(np.matmul(AT, A))
        P = np.matmul(np.matmul(A, inv_ATA), AT)

    elif mode == "ORTHONORMAL":
        Q = base
        QT = np.transpose(Q)
        P = np.matmul(Q, QT)

    return P

def getEquationNormal(a, normal_vector):
    constant = np.matmul(np.transpose(a), normal_vector)
    D = -constant

    plane_equation = np.concatenate([normal_vector, [D]])
    return plane_equation

def centerRotationVector(vector, center, angle):
    ini_vector = vector-center
    ini_vector = np.transpose(np.expand_dims(ini_vector, axis=0))
    v = matrixFromNullspace(ini_vector)
    v1 = v[:, 0] / np.linalg.norm(v[:, 0])
    rot = np.array(np.transpose(ini_vector))
    rot = rot / np.linalg.norm(rot)
    rot = np.concatenate([[np.radians(angle)], rot[0]], axis=0)
    quat = axisToQuaternion(rot)
    quat_rot = rotQuatMatrix(quat)
    v_r = np.matmul(quat_rot, v1)
    v_r = v_r / np.linalg.norm(v_r)
    return v_r

def permutationInvert(matrix):
    def getFirstTuple(t):
        return t[0]

    rows, cols = matrix.shape
    current_matrix = matrix
    permutation_matrix = np.identity(rows)
    rows_ind = rows-cols

    iszero = np.array(matrix == 0).astype(int)
    countzero = np.sum(iszero, axis=0)
    indexes = np.arange(cols)
    zipped_counts = list(zip(countzero, indexes))
    zipped_counts.sort(reverse=True, key=getFirstTuple)
    order = [x[1] for x in zipped_counts]
    print(order)

    for i, inspect_col in enumerate(order):
        #print(inspect_col, i)
        if current_matrix[rows_ind+inspect_col, inspect_col] == 0:
            #print("IS ZERO")
            switched = False
            for j in range(rows):
                #print("J")
                #print(j)
                if not switched and current_matrix[j, inspect_col] != 0 and j not in order[:i-rows_ind]:
                    switched = True
                    switch_matrix = switchPermutation(rows, j, rows_ind+inspect_col)
                    current_matrix = np.matmul(switch_matrix, current_matrix)
                    permutation_matrix = np.matmul(switch_matrix, permutation_matrix)

    return current_matrix, permutation_matrix

def matrixFromNullspace(original_vectors):
    ind_vectors, perm_inv_matrix = permutationInvert(original_vectors)
    reverse_perm_matrix = np.transpose(perm_inv_matrix)
    null_dim = ind_vectors.shape[1]
    full_space_dim = ind_vectors.shape[0]

    identity_shape = full_space_dim-null_dim

    vectors_T = np.transpose(ind_vectors)
    #print(vectors_T)
    #print(vectors_T[:,identity_shape:])
    matrix_invert = np.linalg.inv(vectors_T[:,identity_shape:])

    mod_matrix = np.matmul(matrix_invert, vectors_T)
    T_mod = np.transpose(mod_matrix)

    I = np.identity(identity_shape)
    F = -1*T_mod[0:identity_shape]

    subspace = np.transpose(np.concatenate([I, F], axis=1))
    perm_subspace = np.matmul(reverse_perm_matrix, subspace)
    return perm_subspace

#print(matrixFromNullspace(np.array([[0],[1],[0]])))

#m = np.array([[0],[1],[0],[0]])
#c, p = permutationInvert(m)
#print(m)
#print(c)
#print(p)
def translateH(vectors, trans):
    ones = np.ones([1, vectors.shape[1]])
    appended_vectors = np.concatenate([vectors, ones], axis=0)
    shift_matrix = np.array([[1, 0, 0, trans[0]], [0, 1, 0, trans[1]], [0, 0, 1, trans[2]], [0, 0, 0, 1]])
    shifted = np.matmul(shift_matrix, appended_vectors)
    return shifted[0:3]













