import pygame
from Camera import Camera
from Matrix import *


from Objects import *
import numpy as np

#print(v.value)
width = 500
height = 500
depth = 500

ini_cam_pos, ini_cam_rot = [0,0,0], np.radians([0,0,0])

vectors = np.array([
[0],
[100],
[1000]
])

edges = np.array([
    [0]
])

c = Camera(ini_cam_pos, ini_cam_rot, width, height, depth)
cube = Object3D(Cube(np.array([200,200,200])), [0,0,1000], np.radians(np.array([45,45,0])))
pyramid = Object3D(Pyramid(np.array([300,300,700])), [700,800,2000], np.radians(np.array([0,45,90])))
circle = Object3D(Circle(300, 100), [-500,-500,1500], np.radians(np.array([0,0,0])))
#o = Object3D(Specific(vectors, edges), [0,0,0], np.radians(np.array([0,0,0])))
objects = [cube, pyramid, circle]


def get_edges(vertices, edges):
    rows, cols = edges.shape
    edges_conv = []
    for i in range(rows):
        for j in range(cols):
            if (j < i) and edges[i, j] == 1:
                edges_conv.append((vertices[:, j], vertices[:, i]))

    return edges_conv

def convert_pos(coordinates):
    x, y = coordinates
    new_coordinates = (x+(width/2), y+(height/2))
    return new_coordinates

def join_edges_matrix(m1, m2):
    size1, _ = m1.shape
    size2, _ = m2.shape

    filler = np.zeros((size1, size2))
    upper = np.concatenate([m1, filler], axis=1)
    lower = np.concatenate([np.transpose(filler), m2], axis=1)
    joined = np.concatenate([upper, lower], axis=0)
    return joined

def combine_all_vectors(objects):
    if len(objects)==1:
        return objects[0].getVectors()
    else:
        object_vectors = [x.getVectors() for x in objects]
        final_vectors = np.concatenate(object_vectors, axis=1)
        return final_vectors

def combine_all_edges(objects):
    if len(objects) == 1:
        return objects[0].getEdges()
    else:
        object_vectors = [x.getEdges() for x in objects]
        joint_matrix = join_edges_matrix(object_vectors[0], object_vectors[1])
        for i in range(2, len(object_vectors)):
            joint_matrix = join_edges_matrix(joint_matrix, object_vectors[i])
        return joint_matrix

def post_process(vectors):
    correction_matrix = np.array([[1,0],[0,-1]])
    new_vectors = np.matmul(correction_matrix, vectors)
    return new_vectors


join_edges_matrix(edges, cube.getEdges())


RUNWINDOW = True
DRAW_EDGES = True
DRAW_VERTEX = False

if RUNWINDOW:
    pygame.init()

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Custom Size Window")

    WHITE = pygame.Color("white")
    BLACK = pygame.Color("black")
    GRAY = pygame.Color("gray")
    RED = pygame.Color("red")

    clicking = False
    ini_position = None
    current_position = None

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                clicking = True
                ini_position = np.array(pygame.mouse.get_pos())

            elif event.type == pygame.MOUSEBUTTONUP:
                diff = (current_position - ini_position)
                x_diff, y_diff = post_process(current_position - ini_position)
                #print(x_diff, y_diff)
                c.rotate2DVector(x_diff, y_diff)
                #projected = c.projectVectors(vectors)
                #displayVectors(projected)
                #xy_vectors = c.getXYfromVectors(projected)
                clicking = False


        screen.fill(WHITE)

        if clicking:
            current_position = np.array(pygame.mouse.get_pos())
            pygame.draw.line(screen, BLACK, ini_position, current_position, 2)
            print(vectors)
            print(projected)

        added_vectors = combine_all_vectors(objects)
        projected = c.projectVectors(added_vectors)
        pre_xy_vectors = c.getXYfromVectors(projected)
        xy_vectors = post_process(pre_xy_vectors)
        normal_check = c.checkNormal(added_vectors)

        if DRAW_EDGES:
            added_edges = combine_all_edges(objects)
            edges_conv = get_edges(xy_vectors, added_edges)
            for e in edges_conv:
                _from, _to = e
                pygame.draw.line(screen, BLACK, convert_pos(_from), convert_pos(_to), 1)

        if DRAW_VERTEX:
            xyT = np.transpose(xy_vectors)
            zipped_data = list(zip(xyT[:,0], xyT[:,1], normal_check))
            for x, y, normal in zipped_data:
                if(normal):
                    pygame.draw.circle(screen, BLACK, convert_pos((x, y)), 3)
                else:
                    pygame.draw.circle(screen, GRAY, convert_pos((x, y)), 2)


        pygame.display.update()
        pygame.display.flip()