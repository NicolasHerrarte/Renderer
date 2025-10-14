from Matrix import *
from Discrete import *

class Dynamic3D:
    def __init__(self, object3D, translation_anim, rotation_anim, rotation_center, fps, rotation_type="REGULAR"):
        self.fps = fps

        self.object3D = object3D
        self.vectors = object3D.getVectors()
        self.edges = object3D.getEdges()
        self.rotation_type = rotation_type

        self.translation_anim = translation_anim


        if isinstance(rotation_center, str) and rotation_center == "CENTER":
            self.rotation_center = self.object3D.position
        else:
            self.rotation_center = rotation_center

        if rotation_type == "REGULAR":
            self.rotation_anim_quat = eulerToQuaternion(rotation_anim / self.fps)
        else:
            angles, rotation_angle = rotation_anim
            print(self.object3D.position)
            direction_vector = centerRotationVector(self.object3D.position, rotation_center, rotation_angle)
            axis_vector = np.array([angles / self.fps, direction_vector[0], direction_vector[1], direction_vector[2]])
            self.rotation_anim_quat = axisToQuaternion(axis_vector)

        self.quat_matrix = rotQuatMatrix(self.rotation_anim_quat)
    def rotate(self):
        shifted = translateH(self.vectors, -1 * self.rotation_center)
        rotated = np.matmul(self.quat_matrix, shifted)
        relocated = translateH(rotated, self.rotation_center)
        self.vectors = relocated

    def translate(self):
        shifted = translateH(self.vectors, self.translation_anim)
        self.vectors = shifted

    def getVectors(self):
        return self.vectors

    def getEdges(self):
        return self.edges
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

class Point:
    def __init__(self):
        self.vectors = np.array([[0],[0],[0]])
        self.edges = np.array([[0]])

    def getVertex(self):
        return self.vectors


    def getEdges(self):
        return self.edges

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

        #print(np.transpose([vectors[:, -1]]))
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


class SemiCircle:
    def __init__(self, radius, resolution, order="NORMAL"):
        self.radius = radius
        self.resolution = resolution
        self.order = order

    def getVertex(self):
        vectors = np.array([
            [0],
            [1],
            [0]
        ])

        #print(np.transpose([vectors[:, -1]]))
        euler_rot = [0, 0, math.pi/self.resolution]
        quat = eulerToQuaternion(euler_rot)
        rot_mat = rotQuatMatrix(quat)
        for i in range(self.resolution):
            new_vector = np.matmul(rot_mat, np.transpose([vectors[:, -1]]))
            vectors = np.concatenate([vectors, new_vector], axis=1)

        circle_vectors = vectors*self.radius
        if self.order == "CONVEYOR":
            circle_vectors = conveyorVertex(circle_vectors, 1)
            #print(circle_vectors)
        return circle_vectors

    def getEdges(self):
        S = colShiftMatrix(self.resolution*2, 1)[0:self.resolution+1,0:self.resolution+1]
        if self.order == "CONVEYOR":
            S = conveyorEdges(S, 1)
            #S = overlapHalvesMatrix(S)
            print(S)
        return S

class Sphere:
    def __init__(self, radius, segments, rings):
        self.radius = radius
        self.segments = segments
        self.rings = rings
        self.base = SemiCircle(radius, self.rings, "CONVEYOR")

    def getVertex(self):
        base = self.base.getVertex()
        static = base[:,0:2]
        rotate = base[:,2:]

        print(static.shape)
        print(rotate.shape)
        print(base.shape)
        vector_n = rotate.shape[1]
        print(vector_n)

        euler_rot = [0,(2*math.pi) / self.segments, 0]
        quat = eulerToQuaternion(euler_rot)
        rot_mat = rotQuatMatrix(quat)

        for i in range(self.segments-1):
            new_rotate = np.matmul(rot_mat, rotate[:, vector_n*i:vector_n*(i+1)])
            rotate = np.concatenate([rotate, new_rotate], axis=1)

        return np.concatenate([static, rotate], axis=1)

    def getEdges(self):
        base = self.base.getEdges()
        base_shape = base.shape[0]
        vector_n = base.shape[0]-2
        mold = np.zeros((2+(self.segments)*vector_n, 2+(self.segments)*vector_n))

        bin_vectors = binMatrixToVectors(base)
        bin_vectors[1] = bin_vectors[1]-1

        for i in range(1,self.segments):
            norm_bin = np.array((bin_vectors!=0)).astype(int)*vector_n*i
            added_bin = bin_vectors+norm_bin
            for x, y in np.transpose(added_bin):
                mold[y+1, x] = 1

        horizontal_vertex = colShiftMatrix((self.segments)*vector_n, vector_n)
        mold[2:, 2:] = mold[2:, 2:]+horizontal_vertex
        mold[0:base_shape, 0:base_shape] = base
        mold = overlapHalvesMatrix(mold)
        return mold

class Cone:
    def __init__(self, radius, rings, height):
        self.radius = radius
        self.rings = rings
        self.height = height
        self.base = Circle(radius, self.rings)

    def getVertex(self):
        base = self.base.getVertex()
        base = np.concatenate([base, [[0],[0],[self.height]]], axis=1)
        return base

    def getEdges(self):
        base = self.base.getEdges()
        top_edges = np.ones([1, self.rings])
        cone_edges = np.concatenate([base, top_edges], axis=0)
        padding = np.zeros([self.rings+1, 1])
        padded_edges = np.concatenate([cone_edges, padding], axis=1)
        return padded_edges

class Cylinder:
    def __init__(self, radius, rings, height):
        self.radius = radius
        self.rings = rings
        self.height = height
        self.base = Circle(radius, self.rings)

    def getVertex(self):
        base1 = self.base.getVertex()
        base2 = self.base.getVertex()
        base2[-1] += self.height
        bases = np.concatenate([base1, base2], axis=1)
        return bases

    def getEdges(self):
        base1 = self.base.getEdges()
        print(base1.shape)
        base2 = self.base.getEdges()
        filler = np.identity(self.rings)
        padding = np.zeros(base1.shape)

        upper = np.concatenate([base1, filler], axis=1)
        lower = np.concatenate([padding, base2], axis=1)

        full = np.concatenate([upper, lower], axis=0)
        overlapped = overlapHalvesMatrix(full)
        return overlapped

cylinder = Cylinder(1, 6, 1)
cylinder.getVertex()
cylinder.getEdges()

#





