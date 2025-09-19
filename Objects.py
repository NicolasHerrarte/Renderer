from Matrix import *

class Specific:
    def __init__(self, vectors, edges):
        self.vectors = vectors
        self.edges = edges

    def getVectors(self):
        return self.vectors

    def getEdges(self):
        return self.edges


class Cube:
    def __init__(self, size, position, euler_rotation):
        self.size = size
        self.position = position
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
        #self.euler_rotation = euler_rotation
        #self.quat_rotation = eulerToQuaternion(self.euler_rotation)

    def getCorners(self):
        width, height, depth = self.size
        sizes_matrix = np.array([[width,0,0],[0,height,0],[0,0,depth]])
        perm = permMatrix([1, -1], 3)
        coordinates = 0.5*np.matmul(sizes_matrix, perm)
        return coordinates

    def moveCorners(self):
        vectors = self.getCorners()

        rows, cols = vectors.shape
        for c in range(cols):
            vectors[:, c] = vectors[:, c] + self.position

        return vectors

    def getVectors(self):
        return self.moveCorners()

    def getEdges(self):
        return self.regular_grid

class Pyramid:
    def __init__(self, size, position, euler_rotation):
        self.size = size
        self.position = position
        self.regular_grid = np.array([
            [0,0,0,0,0],
            [1,0,0,0,0],
            [1,0,0,0,0],
            [0,1,1,0,0],
            [1,1,1,1,0],
        ])
        #self.euler_rotation = euler_rotation
        #self.quat_rotation = eulerToQuaternion(self.euler_rotation)

    def getCorners(self):
        width, height, depth = self.size
        sizes_matrix = np.array([[width,0,0],[0,height,0],[0,0,depth]])
        perm = permMatrix([1, -1], 3)
        filtered = np.transpose(np.array([x for x in np.transpose(perm) if x[2] == -1]))
        filtered_perm = np.concatenate([filtered, np.array([[0], [0], [1]])], axis=1)
        coordinates = 0.5*np.matmul(sizes_matrix, filtered_perm)
        return coordinates

    def moveCorners(self):
        vectors = self.getCorners()

        rows, cols = vectors.shape
        for c in range(cols):
            vectors[:, c] = vectors[:, c] + self.position

        return vectors

    def getVectors(self):
        return self.moveCorners()

    def getEdges(self):
        return self.regular_grid

cube = Cube([2,2,2], [0,0,0], 1)
print(cube.getCorners())

