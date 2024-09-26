import numpy as np
from itertools import product
from .normalization import normalize_edf
from .basis import legendre_basis

def estimate_hcr_coefficients_nd(data: np.ndarray, m: int) -> np.ndarray:
    """Estimate HCR coefficients for n-dimensional data."""
    n, d = data.shape
    normalized_data = np.array([normalize_edf(data[:, i]) for i in range(d)]).T
    
    basis = [legendre_basis(m, normalized_data[:, i]) for i in range(d)]
    
    coefficients = np.zeros([m+1] * d)
    for idx in product(range(m+1), repeat=d):
        coefficients[idx] = np.mean(np.prod([basis[i][idx[i]] for i in range(d)], axis=0))
    
    return coefficients