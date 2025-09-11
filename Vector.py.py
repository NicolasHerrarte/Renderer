import numpy as np
import pygame

class Vector:
    def __init__(self, dimension, value=None):
        self.dimension = dimension
        if value is None:
            self.value = np.zeros((self.dimension, 1))
        else:
            self.value = value
