import numpy as np
from scipy import stats
from sklearn.metrics import mutual_info_score
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split as sk_train_test_split
from typing import Tuple

def normalize_gaussian(data: np.ndarray) -> np.ndarray:
    """
    Normalize data to ~uniform[0,1] using Gaussian CDF.
    Args:
        data (np.ndarray): Input data to be normalized.
    Returns:
        np.ndarray: Normalized data.
    """
    mean = np.mean(data)
    std = np.std(data)
    return stats.norm.cdf((data - mean) / std)

def normalize_edf(data: np.ndarray) -> np.ndarray:
    """
    Normalize data to ~uniform[0,1] using Empirical Distribution Function.
    Args:
        data (np.ndarray): Input data to be normalized.
    Returns:
        np.ndarray: Normalized data.
    """
    sorted_data = np.sort(data)
    ranks = np.argsort(data)
    n = len(data)
    edf = (ranks + 0.5) / n
    return edf

def compute_mutual_information(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute mutual information between X and Y using a histogram-based approach.
    
    Args:
        X (np.ndarray): First variable.
        Y (np.ndarray): Second variable.
    
    Returns:
        float: Estimated mutual information.
    """
    X_discrete = np.digitize(X, np.linspace(X.min(), X.max(), 20))
    Y_discrete = np.digitize(Y, np.linspace(Y.min(), Y.max(), 20))
    
    return mutual_info_score(X_discrete, Y_discrete)

def train_test_split(X: np.ndarray, y: np.ndarray, test_size: float = 0.2, random_state: int = None) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split the data into training and testing sets.
    Args:
        X (np.ndarray): Input features.
        y (np.ndarray): Target values.
        test_size (float): Proportion of the data to include in the test split.
        random_state (int): Seed for random number generator.
    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: X_train, X_test, y_train, y_test
    """
    return sk_train_test_split(X, y, test_size=test_size, random_state=random_state)

def load_mnist():
    """
    Load and return the MNIST dataset using sklearn.
    
    Returns:
        tuple: (x_train, y_train), (x_test, y_test)
    """
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X, y = mnist.data, mnist.target.astype(int)
    
    # Split the data into train and test sets
    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    return (x_train, y_train), (x_test, y_test)

def preprocess_data(x_train, x_test):
    """
    Preprocess the input data by normalizing.
    
    Args:
        x_train (np.ndarray): Training data
        x_test (np.ndarray): Test data
    
    Returns:
        tuple: Preprocessed (x_train, x_test)
    """
    # Normalize
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_test = scaler.transform(x_test)
    
    return x_train, x_test

def one_hot_encode(y, num_classes=10):
    """
    Convert integer labels to one-hot encoded labels.
    
    Args:
        y (np.ndarray): Input labels
        num_classes (int): Number of classes
    
    Returns:
        np.ndarray: One-hot encoded labels
    """
    return np.eye(num_classes)[y]

def accuracy_score(y_true, y_pred):
    """
    Calculate the accuracy score.
    
    Args:
        y_true (np.ndarray): True labels
        y_pred (np.ndarray): Predicted labels
    
    Returns:
        float: Accuracy score
    """
    return np.mean(y_true == y_pred)