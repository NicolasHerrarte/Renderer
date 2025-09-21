from Matrix import *

CAMERA_DEFAULT_BASE = "REGULAR"
class Camera:
    def __init__(self, origin, euler_rotation, width, height, depth):
        self.origin = origin
        self.width = width
        self.height = height
        self.depth = depth
        self.euler_rotation = euler_rotation
        self.quat_rotation = eulerToQuaternion(self.euler_rotation)

    def getVectorsBase(self, mode):
        borders = self.getCorners()
        a = borders[:, 0]
        b = borders[:, 1]
        c = borders[:, 2]

        u = np.expand_dims(a - b, axis=0)
        v = np.expand_dims(a - c, axis=0)
        A = np.transpose(np.concatenate([u, v], axis=0))

        if mode == "REGULAR":
            return A
        elif mode == "ORTHONORMAL":
            Q, _  = np.linalg.qr(A)
            return Q

    def projectionMatrix(self, mode=CAMERA_DEFAULT_BASE):
        if mode == "REGULAR":
            A = self.getVectorsBase(mode)
            AT = np.transpose(A)
            inv_ATA = np.linalg.inv(np.matmul(AT, A))
            P = np.matmul(np.matmul(A, inv_ATA), AT)

        elif mode == "ORTHONORMAL":
            Q = self.getVectorsBase(mode)
            QT = np.transpose(Q)
            P = np.matmul(Q, QT)

        return P

    def errorMatrix(self, mode=CAMERA_DEFAULT_BASE):
        P = self.projectionMatrix(mode)
        return np.subtract(np.identity(P.shape[0]), P)

    def checkNormal(self, vectors_original):
        vectors = np.copy(vectors_original)
        norm = self.normalVector()
        E = self.errorMatrix()
        errors = np.matmul(E, vectors)
        rows, cols = vectors.shape
        constants = np.zeros([cols])
        for c in range(cols):
            if norm[0] != 0:
                constants[c] = errors[0, c]/norm[0]
            elif norm[1] != 0:
                constants[c] = errors[1, c] / norm[1]
            elif norm[2] != 0:
                constants[c] = errors[2, c] / norm[2]

        #displayVectors(errors, special=True)
        boolean_arr = np.array((constants>=0))
        return boolean_arr



    def normalVector(self, normal=True):
        A = self.getVectorsBase("REGULAR")
        u = A[:,0]
        v = A[:,1]
        cross_vector = np.cross(v, u)
        if not normal:
            return cross_vector
        else:
            normal_vector = cross_vector / np.linalg.norm(cross_vector)
            return normal_vector

    def projectVectors(self, vectors_original):
        vectors = np.copy(vectors_original)
        norm = self.normalVector()
        borders = self.getCorners()

        depth_vector = -norm*self.depth
        #displayVectors(np.transpose([depth_vector]), special=True)
        rows, cols = vectors.shape
        #displayVectors(vectors)
        for c in range(4):
            borders[:, c] = borders[:, c]-depth_vector
        for c in range(cols):
            vectors[:, c] = vectors[:, c]-depth_vector

        #displayVectors(np.transpose([depth_vector]))
        x = np.matmul(np.transpose(norm), vectors)
        constant = np.matmul(borders[:,0], norm)
        D = -constant
        t = -np.reciprocal(x)*D
        for c in range(cols):
            vectors[:, c] = (vectors[:, c]*t[c])+depth_vector
        return vectors


    def getEquation(self):
        borders = self.getCorners()
        a = borders[:, 0]
        normal_vector = self.normalVector()
        constant = np.matmul(np.transpose(a), normal_vector)
        D = -constant

        plane_equation = np.concatenate([normal_vector, [D]])
        return plane_equation

    def getXYfromVectors(self, vectors, singular=False):
        if not singular:
            ones = np.expand_dims(np.ones(vectors.shape[1]), axis=0)
            extended_vectors = np.concatenate([vectors, ones], axis=0)
            plane_equation = self.getEquation()
            equality = np.matmul(plane_equation, extended_vectors)
            #print(equality)
            #assert np.all(np.absolute(equality) < EQUALITY_NUM)
            inv_quat_rot = quatInv(self.quat_rotation)
            inv_quat_mat = rotQuatMatrix(inv_quat_rot)
            relative_vectors = np.matmul(inv_quat_mat, vectors)
            return relative_vectors[0:-1]
        else:
            x, y, z = vectors
            vector_extended = [x, y, z, 1]
            plane_equation = self.getEquation()
            equality = np.matmul(np.transpose(vector_extended), plane_equation)
            #assert abs(equality) < EQUALITY_NUM
            inv_quat_rot = quatInv(self.quat_rotation)
            inv_quat_mat = rotQuatMatrix(inv_quat_rot)
            relative_vector = np.matmul(inv_quat_mat, vectors)
            #print(relative_vector)
            x, y, _ = relative_vector
            return x, y

    def rotate2DVector(self, x, y):
        x = x/self.width
        y = y/self.width

        #print(x)
        #print(y)
        normal_vector = self.normalVector()
        ground_vector = np.array([x, y, 0])
        relative_vector = self.getPointsOnPlane(np.transpose(ground_vector))
        #print(relative_vector)
        reference_vector = relative_vector + normal_vector

        axis_vector = np.cross(normal_vector, reference_vector)
        axis_vector = axis_vector / np.linalg.norm(axis_vector)

        numerator = np.matmul(np.transpose(normal_vector), reference_vector)
        denominator = (np.linalg.norm(normal_vector) * np.linalg.norm(reference_vector))
        theta = np.arccos(numerator / denominator)

        axis_rotation = np.concatenate([[theta], axis_vector])
        quat = axisToQuaternion(axis_rotation)
        self.quat_rotation = quatMult(quat, self.quat_rotation)
        self.euler_rotation = quaternionToEuler(quat)

    def getPointsOnPlane(self, points):
        quat = self.quat_rotation
        R = rotQuatMatrix(quat)
        rotated_points = np.matmul(R, points)
        return rotated_points

    def getCorners(self):
        borders = 0.5 * np.matmul(np.array([[self.width, 0], [0, self.height], [0, 0]]), permMatrix([1, -1], 2))
        return self.getPointsOnPlane(borders)


width = 2
height = 2
depth = 1

ini_cam_pos, ini_cam_rot = np.array([0,0,0]), np.radians(np.array([0,0,0]))

c = Camera(ini_cam_pos, ini_cam_rot, width, height, depth)
vectors = np.transpose(np.array([[3.0,2.0, -10.0],[4.0, -2.0, 10.0]]))
c.rotate2DVector(-0.5, -0.5)
c.checkNormal(vectors)
p = c.projectVectors(vectors)
corners = c.getCorners()
e = c.getEquation()
#displayVectors(p, special=True)
#displayVectors(corners, special=True)
#displayVectors(np.transpose([e]))
xy = c.getXYfromVectors(p)

#displayVectors(p)
#displayVectors(xy)
