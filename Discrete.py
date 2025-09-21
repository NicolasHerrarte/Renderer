import numpy as np
from Matrix import *

def conveyorVertex(vertex, pos):
    new_vertex = np.delete(np.insert(vertex, pos, vertex[:,-1], axis=1), -1, axis=1)
    return new_vertex

def conveyorEdges(edges_original, pos):
    edges = np.copy(edges_original)
    edges[pos, pos - 1] = 0
    edges[pos + 1, pos - 1] = 1
    edges[pos + 1, pos] = 0
    edges[pos, -1] = 1

    return edges
