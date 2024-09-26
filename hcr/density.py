from numba import jit
import numpy as np
from itertools import product
from .basis import legendre_basis

@jit(nopython=True)
def hcr_density_nd(x: np.ndarray, coefficients: np.ndarray) -> float:
    """Compute HCR density for given n-dimensional coefficients."""
    d = len(x)
    m = coefficients.shape[0] - 1
    basis = [legendre_basis(m, xi) for xi in x]
    
    density = 1.0
    for idx in product(range(1, m+1), repeat=d):
        density += coefficients[idx] * np.prod([basis[i][idx[i]] for i in range(d)])
    
    return density