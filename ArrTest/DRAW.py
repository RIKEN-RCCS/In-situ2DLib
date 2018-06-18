import numpy as np
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

def draw(dims, Z):
    x = np.linspace(0, 1, dims[0])
    y = np.linspace(0, 1, dims[1])
 
    X, Y = np.meshgrid(x, y)
    #Z = np.sqrt(X**2 + Y**2)
 
    plt.contour(X, Y, Z)
 
    plt.gca().set_aspect('equal')
    plt.savefig('X.png')

    return 1
