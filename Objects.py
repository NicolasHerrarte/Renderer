from Matrix import *

class Object3D:
    def __init__(self, generator, position, euler_rotation):
        self.vectors = generator.getVertex()
        self.edges = generator.getEdges()
        self.position = position
        self.euler_rotation = euler_rotation
        self.quat_rotation = eulerToQuaternion(self.euler_rotation)

    def rotateCorners(self, vectors):
        rot_matrix = rotQuatMatrix(self.quat_rotation)
        rotated_vectors = np.matmul(rot_matrix, vectors)

        return rotated_vectors

    def moveCorners(self, vectors):
        rows, cols = vectors.shape
        for c in range(cols):
            vectors[:, c] = vectors[:, c] + self.position

        return vectors

    def getVectors(self):
        rot_vectors = self.rotateCorners(self.vectors)
        pos_vectors = self.moveCorners(rot_vectors)
        return pos_vectors

    def getEdges(self):
        return self.edges


class Cube:
    def __init__(self, size):
        self.size = size
        self.regular_grid = np.array([
            [0,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0],
            [1,0,0,0,0,0,0,0],
            [0,1,1,0,0,0,0,0],
            [1,0,0,0,0,0,0,0],
            [0,1,0,0,1,0,0,0],
            [0,0,1,0,1,0,0,0],
            [0,0,0,1,0,1,1,0]
        ])

    def getVertex(self):
        width, height, depth = self.size
        sizes_matrix = np.array([[width,0,0],[0,height,0],[0,0,depth]])
        perm = permMatrix([1, -1], 3)
        coordinates = 0.5*np.matmul(sizes_matrix, perm)
        return coordinates

    def getEdges(self):
        return self.regular_grid

class Pyramid:
    def __init__(self, size):
        self.size = size
        self.regular_grid = np.array([
            [0,0,0,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0],
            [0,1,1,0,0],
            [1,1,1,1,0],
        ])

    def getVertex(self):
        width, height, depth = self.size
        sizes_matrix = np.array([[width,0,0],[0,height,0],[0,0,depth]])
        perm = permMatrix([1, -1], 3)
        filtered = np.transpose(np.array([x for x in np.transpose(perm) if x[2] == -1]))
        filtered_perm = np.concatenate([filtered, np.array([[0], [0], [1]])], axis=1)
        coordinates = 0.5*np.matmul(sizes_matrix, filtered_perm)
        return coordinates


    def getEdges(self):
        return self.regular_grid

class Specific:
    def __init__(self, vectors, edges):
        self.vectors = vectors
        self.edges = edges

    def getVertex(self):
        return self.vectors

    def getEdges(self):
        return self.edges

class Circle:

    def __init__(self, radius, resolution):
        self.radius = radius
        self.resolution = resolution

    def getVertex(self):
        vectors = np.array([
            [0],
            [1],
            [0]
        ])

        print(np.transpose([vectors[:, -1]]))
        euler_rot = [0, 0, (2 * math.pi) / self.resolution]
        quat = eulerToQuaternion(euler_rot)
        rot_mat = rotQuatMatrix(quat)
        for i in range(self.resolution-1):
            new_vector = np.matmul(rot_mat, np.transpose([vectors[:, -1]]))
            vectors = np.concatenate([vectors, new_vector], axis=1)

        return vectors*self.radius

    def getEdges(self):
        S = colShiftMatrix(self.resolution, 1)
        return overlapHalvesMatrix(S)


