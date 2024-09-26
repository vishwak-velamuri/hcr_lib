import numpy as np
from scipy.special import legendre
from numba import jit

@jit(nopython=True)
def legendre_basis(m: int, x: np.ndarray) -> np.ndarray:
    """Generate orthonormal Legendre polynomial basis up to degree m."""
    basis = np.zeros((m + 1, len(x)))
    for i in range(m + 1):
        poly = legendre(i)
        basis[i] = np.sqrt(2*i + 1) * poly(2*x - 1)
    return basis