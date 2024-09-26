import numpy as np
from scipy import stats

def normalize_gaussian(data: np.ndarray) -> np.ndarray:
    """Normalize data to ~uniform[0,1] using Gaussian CDF."""
    mean = np.mean(data)
    std = np.std(data)
    return stats.norm.cdf((data - mean) / std)

def normalize_edf(data: np.ndarray) -> np.ndarray:
    """Normalize data to ~uniform[0,1] using Empirical Distribution Function."""
    sorted_data = np.sort(data)
    ranks = np.argsort(data)
    n = len(data)
    edf = (ranks + 0.5) / n
    return edf