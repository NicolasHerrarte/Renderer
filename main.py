import pygame
import numpy

x_width = 600
y_width = 400

x_width = 600
y_width = 400

pygame.init()

screen = pygame.display.set_mode((x_width, y_width))
pygame.display.set_caption("Custom Size Window")

WHITE = pygame.Color("white")

running = True
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    screen.fill(WHITE)

    pygame.display.flip()