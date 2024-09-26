import matplotlib.pyplot as plt
import numpy as np
from .density import hcr_density_nd

def plot_hcr_density_2d(coefficients: np.ndarray, m: int, num_points: int = 100):
    """Plot the HCR density for 2D data."""
    x = np.linspace(0, 1, num_points)
    y = np.linspace(0, 1, num_points)
    X, Y = np.meshgrid(x, y)
    
    Z = np.zeros((num_points, num_points))
    for i in range(num_points):
        for j in range(num_points):
            Z[i, j] = hcr_density_nd([X[i, j], Y[i, j]], coefficients)
    
    plt.figure(figsize=(10, 8))
    plt.contourf(X, Y, Z, levels=20, cmap='viridis')
    plt.colorbar(label='Density')
    plt.title("2D HCR Density")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.show()