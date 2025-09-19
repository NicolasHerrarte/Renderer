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

def displayVectors(vectors, special=False):
    vectors = np.transpose(vectors)
    for v in vectors:
        v = [str(x) for x in v]
        if special:
            print(Fore.BLUE+"("+",".join(v)+")"+Style.RESET_ALL)
        else:
            print("(" + ",".join(v) + ")")














