import pygame
from Camera import Camera
import numpy as np

#print(v.value)
width = 500
height = 500
depth = 500

ini_cam_pos, ini_cam_rot = [0,0,0], [0,0,0]

c = Camera(ini_cam_pos, ini_cam_rot, width, height, depth)
vectors = np.array([
[0],
[0],
[1000]])
RUNWINDOW = True

if RUNWINDOW:
    pygame.init()

    screen = pygame.display.set_mode((width, height))
    pygame.display.set_caption("Custom Size Window")

    WHITE = pygame.Color("white")
    BLACK = pygame.Color("black")

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
                x_diff, y_diff = (current_position - ini_position)
                c.rotate2DVector(x_diff, y_diff)
                projected = c.projectVectors(vectors)
                print(projected)
                clicking = False

        screen.fill(WHITE)

        if clicking:
            current_position = np.array(pygame.mouse.get_pos())
            pygame.draw.line(screen, BLACK, ini_position, current_position, 2)
            #print(current_position - ini_position)


        pygame.draw.circle(screen, BLACK, (0, 0), 5)
        projected = c.projectVectors(vectors)
        #print(projected)
        for x, y in np.transpose(projected):
            pygame.draw.circle(screen, BLACK, (x+(width/2), y+(height/2)), 5)

        pygame.display.update()
        pygame.display.flip()