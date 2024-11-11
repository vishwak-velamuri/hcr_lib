import numpy as np
from scipy import stats
import yfinance as yf
from datetime import datetime
from typing import Tuple, Union, Optional

def get_financial_data():
    start_date = datetime(2000, 1, 1)
    dji = yf.download('^DJI', start=start_date)
    return dji['Close'].values

def calculate_log_returns(prices: np.ndarray, dimension: int = 1) -> np.ndarray:
    """
    Calculate log returns and create d-dimensional vectors.
    
    Args:
        prices: Array of prices
        dimension: Number of consecutive returns to group together
    
    Returns:
        Array of d-dimensional log return vectors
    """
    # Calculate log returns
    log_returns = np.log(prices[1:] / prices[:-1])
    
    # For 1D, return as is
    if dimension == 1:
        return log_returns.reshape(-1, 1)
    
    # For higher dimensions, create vectors using a sliding window
    n = len(log_returns) - dimension + 1
    vectors = np.zeros((n, dimension))
    for i in range(n):
        vectors[i] = log_returns[i:i+dimension]
    return vectors

def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Normalize data using Student's t-distribution fit.
    
    Args:
        data: Input data array of shape (n_samples, n_dimensions)
    
    Returns:
        Normalized data array of same shape
    """
    normalized = np.zeros_like(data)
    
    # Normalize each dimension independently
    for i in range(data.shape[1]):
        params = stats.t.fit(data[:, i])
        normalized[:, i] = stats.t.cdf(data[:, i], *params)
    
    return normalized

def split_train_test(data: np.ndarray, train_ratio: float = 0.5, random_state: Optional[int] = None) -> Tuple[np.ndarray, np.ndarray]:
    """
    Split data into training and testing sets.
    
    Args:
        data: Input data array
        train_ratio: Ratio of training data
        random_state: Random seed for reproducibility
    
    Returns:
        Tuple of (training_data, testing_data)
    """
    if random_state is not None:
        np.random.seed(random_state)
    
    indices = np.random.permutation(len(data))
    split_idx = int(len(data) * train_ratio)
    
    return data[indices[:split_idx]], data[indices[split_idx:]]

def compute_kde_log_likelihood(data: np.ndarray, dimension, bandwidth: Union[str, float, np.ndarray] = 'scott', train_ratio: float = 0.5, random_state: Optional[int] = None) -> float:
    """
    Compute KDE and return log-likelihood for d-dimensional data.
    
    Args:
        data: Input price data
        dimension: Number of dimensions for KDE
        bandwidth: Bandwidth method or value for KDE
        train_ratio: Ratio of training data
        random_state: Random seed for reproducibility
    
    Returns:
        Log-likelihood value
    """
    # Calculate log returns and create d-dimensional vectors
    log_returns = calculate_log_returns(data, dimension)
    
    # Normalize the data
    normalized_data = normalize_data(log_returns)
    
    # Split into training and testing sets
    train_data, test_data = split_train_test(
        normalized_data,
        train_ratio=train_ratio,
        random_state=random_state
    )
    
    # Fit KDE on training data
    kde = stats.gaussian_kde(
        train_data.T,
        bw_method=bandwidth
    )
    
    # Calculate log-likelihood on test data
    log_likelihood = np.mean(np.log(kde(test_data.T)))
    
    return log_likelihood

def main():
    # Get data
    data = get_financial_data()
    
    # Test different dimensions
    dimensions = [1, 2, 3, 4]

    for dim in dimensions:
        log_likelihood = compute_kde_log_likelihood(data, dimension=dim, random_state=42)
        print(f"{dim}D KDE log-likelihood: {log_likelihood:.6f}")

if __name__ == "__main__":
    main()