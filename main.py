import pygame
from Camera import Camera
from Matrix import *
from Objects import *
import numpy as np

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

def post_process(vectors, invert_x, invert_y):
    _x = 1
    _y = 1
    if invert_x:
        _x *= -1

    if invert_y:
        _y *= -1
    correction_matrix = np.array([[_x,0],[0,_y]])
    new_vectors = np.matmul(correction_matrix, vectors)
    return new_vectors


RUNWINDOW = True
NEWPROYECTION = True
DRAW_EDGES = True
DRAW_VERTEX = True

CAMERA_SPEED = 5
FPS = 60
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

c = Camera(ini_cam_pos, ini_cam_rot, CAMERA_SPEED, width, height, depth)
cube = Object3D(Cube(np.array([200,200,200])), np.array([0,500,2000]), np.radians(np.array([45,45,0])))
pyramid = Object3D(Pyramid(np.array([300,300,500])), np.array([700,800,2000]), np.radians(np.array([0,0,0])))
circle = Object3D(SemiCircle(300, 4, order="CONVEYOR"), np.array([-500,-500,1500]), np.radians(np.array([0,0,0])))
sphere = Object3D(Sphere(200, 10, 10), np.array([0,0,2000]), np.radians(np.array([0,0,0])))
cone = Object3D(Cone(200, 10, 500), np.array([600,-600,1500]), np.radians(np.array([0,30,0])))
cylinder = Object3D(Cylinder(300, 6, 500), np.array([-750,800,2000]), np.radians(np.array([0,-45,0])))
point = Object3D(Point(), np.array([200,0,1000]), np.radians(np.array([0,0,0])))

dynamic_cube = Dynamic3D(cube, np.array([5,0,0]), np.radians([35,45]), np.array([0,0,2000]), FPS, rotation_type="ORBIT")
dynamic_cone = Dynamic3D(cone, np.array([0,0,0]), np.radians(np.array([0,30,15])), "CENTER", FPS)
dynamic_cylinder = Dynamic3D(cylinder, np.array([0,0,0]), np.radians(np.array([10,30,50])), "CENTER", FPS)
dynamic_sphere = Dynamic3D(sphere, np.array([0,0,0]), np.radians(np.array([45,30,20])), "CENTER", FPS)

objects = [dynamic_cube, dynamic_sphere]
dynamics = [dynamic_cube, dynamic_sphere]

added_vectors = combine_all_vectors(objects)

if NEWPROYECTION:
    pre_xy_vectors = c.proyectVectorsV2(added_vectors)
    xy_vectors = post_process(pre_xy_vectors, True, False)
else:
    projected = c.projectVectors(added_vectors)
    pre_xy_vectors = c.getXYfromVectors(projected)
    xy_vectors = post_process(pre_xy_vectors, False, True)

normal_check = c.checkNormal(added_vectors)

#print("AAA")
#print(xy_vectors.shape)
#print(v2proyected.shape)

if RUNWINDOW:
    pygame.init()
    clock = pygame.time.Clock()

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

        for d in dynamics:
            d.rotate()
            d.translate()

        added_vectors = combine_all_vectors(objects)

        if NEWPROYECTION:
            pre_xy_vectors = c.proyectVectorsV2(added_vectors)
            xy_vectors = post_process(pre_xy_vectors, True, False)
        else:
            projected = c.projectVectors(added_vectors)
            pre_xy_vectors = c.getXYfromVectors(projected)
            xy_vectors = post_process(pre_xy_vectors, False, True)

        normal_check = c.checkNormal(added_vectors)
        added_vectors = combine_all_vectors(objects)

        keys = pygame.key.get_pressed()
        w_pressed = keys[pygame.K_w]
        s_pressed = keys[pygame.K_s]
        a_pressed = keys[pygame.K_a]
        d_pressed = keys[pygame.K_d]
        up_arrow_pressed = keys[pygame.K_UP]
        down_arrow_pressed = keys[pygame.K_DOWN]

        if w_pressed:
            c.move("UP")
        if s_pressed:
            c.move("DOWN")
        if a_pressed:
            c.move("LEFT")
        if d_pressed:
            c.move("RIGHT")
        if up_arrow_pressed:
            c.move("FORWARD")
        if down_arrow_pressed:
            c.move("BACKWARD")

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            if event.type == pygame.MOUSEBUTTONDOWN:
                clicking = True
                ini_position = np.array(pygame.mouse.get_pos())

            elif event.type == pygame.MOUSEBUTTONUP:
                diff = (current_position - ini_position)
                if NEWPROYECTION:
                    x_diff, y_diff = post_process(current_position - ini_position, True, False)
                else:
                    x_diff, y_diff = post_process(current_position - ini_position, False, True)
                c.rotate2DVector(x_diff, y_diff)


                #displayVectors(v2proyected)
                #displayVectors(xy_vectors)
                clicking = False


        screen.fill(WHITE)

        if clicking:
            current_position = np.array(pygame.mouse.get_pos())
            pygame.draw.line(screen, BLACK, ini_position, current_position, 2)
            #print(vectors)
            #print(projected)


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

        dt = clock.tick(FPS)
        pygame.display.update()
        pygame.display.flip()