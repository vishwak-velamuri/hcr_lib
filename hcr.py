import numpy as np
from scipy import stats
from scipy.special import legendre
from itertools import product
from typing import Callable

class HCR:
    def __init__(self, m: int):
        """
        Initialize the HCR (Hermite-like Continuous Representation) object.

        Args:
            m (int): Maximum degree of Legendre polynomials.
        """
        self.m = m
        self.coefficients = None

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def legendre_basis(m: int, x: np.ndarray) -> np.ndarray:
        """
        Generate orthonormal Legendre polynomial basis up to degree m.

        Args:
            m (int): Maximum degree of Legendre polynomials.
            x (np.ndarray): Input values.

        Returns:
            np.ndarray: Orthonormal Legendre polynomial basis.
        """
        basis = np.zeros((m + 1, len(x)))
        for i in range(m + 1):
            poly = legendre(i)
            basis[i] = np.sqrt(2*i + 1) * poly(2*x - 1)
        return basis

    def estimate_coefficients_1d(self, data: np.ndarray) -> np.ndarray:
        """
        Estimate HCR coefficients for 1D data.

        Args:
            data (np.ndarray): Input 1D data.

        Returns:
            np.ndarray: Estimated HCR coefficients.
        """
        normalized_data = self.normalize_edf(data)
        basis = self.legendre_basis(self.m, normalized_data)
        self.coefficients = np.mean(basis, axis=1)
        return self.coefficients

    def estimate_coefficients_nd(self, data: np.ndarray) -> np.ndarray:
        """
        Estimate HCR coefficients for n-dimensional data.

        Args:
            data (np.ndarray): Input n-dimensional data.

        Returns:
            np.ndarray: Estimated HCR coefficients.
        """
        n, d = data.shape
        normalized_data = np.array([self.normalize_edf(data[:, i]) for i in range(d)]).T
        
        basis = [self.legendre_basis(self.m, normalized_data[:, i]) for i in range(d)]
        
        self.coefficients = np.zeros([self.m+1] * d)
        for idx in product(range(self.m+1), repeat=d):
            self.coefficients[idx] = np.mean(np.prod([basis[i][idx[i]] for i in range(d)], axis=0))
        
        return self.coefficients

    def density_1d(self, x: np.ndarray) -> np.ndarray:
        """
        Compute HCR density for given 1D coefficients.

        Args:
            x (np.ndarray): Input values.

        Returns:
            np.ndarray: Computed HCR density.
        """
        basis = self.legendre_basis(self.m, x)
        return 1 + np.dot(self.coefficients[1:], basis[1:])

    def density_nd(self, x: np.ndarray) -> float:
        """
        Compute HCR density for given n-dimensional coefficients.

        Args:
            x (np.ndarray): Input values.

        Returns:
            float: Computed HCR density.
        """
        d = len(x)
        basis = [self.legendre_basis(self.m, xi) for xi in x]
        
        density = 1.0
        for idx in product(range(1, self.m+1), repeat=d):
            density += self.coefficients[idx] * np.prod([basis[i][idx[i]] for i in range(d)])
        
        return density

    def conditional_distribution(self, x: int, fixed_vars: dict) -> Callable:
        """
        Compute conditional distribution for given fixed variables.

        Args:
            x (int): Index of the variable to condition on.
            fixed_vars (dict): Dictionary of fixed variables and their values.

        Returns:
            Callable: Function representing the conditional distribution.
        """
        d = len(self.coefficients.shape)
        
        cond_coeffs = np.zeros_like(self.coefficients)
        for idx in product(range(self.m+1), repeat=d):
            if all(idx[i] == 0 for i in fixed_vars):
                cond_coeffs[idx] = self.coefficients[idx] * np.prod([self.legendre_basis(self.m, fixed_vars[i])[idx[i]] for i in range(d) if i not in fixed_vars])
        
        return lambda y: self.density_nd([y if i not in fixed_vars else fixed_vars[i] for i in range(d)])

    def expected_value(self, var_index: int) -> float:
        """
        Compute expected value for a given variable.

        Args:
            var_index (int): Index of the variable to compute expected value for.

        Returns:
            float: Computed expected value.
        """
        d = len(self.coefficients.shape)
        
        ev = 0.5  # Expected value of uniform distribution on [0, 1]
        for idx in product(range(1, self.m+1), repeat=d):
            if idx[var_index] == 1 and sum(idx) == 1:
                ev += self.coefficients[idx] / np.sqrt(3)  # Integral of x * first Legendre polynomial
        
        return ev

    def transform_basis(self, new_m: int, x: np.ndarray) -> np.ndarray:
        """
        Transform HCR coefficients from one basis size to another.

        Args:
            new_m (int): New maximum degree for transformation.
            x (np.ndarray): Input values.

        Returns:
            np.ndarray: Transformed HCR coefficients in the new basis.
        """
        # Generate original basis
        original_basis = self.legendre_basis(self.m, x)
        
        # Project coefficients onto the original basis
        original_projection = np.dot(self.coefficients, original_basis)
        
        # Generate new basis
        new_basis = self.legendre_basis(new_m, x)
        
        # Recompute coefficients for the new basis
        new_coefficients = np.linalg.lstsq(new_basis.T, original_projection, rcond=None)[0]
        
        return new_coefficients

class HCRRegressor:
    def __init__(self, m: int = 5):
        self.hcr = HCR(m)
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit the HCR regressor to the training data.

        Args:
            X (np.ndarray): Training input samples.
            y (np.ndarray): Target values.

        Returns:
            self: Returns an instance of self.
        """
        data = np.column_stack((X, y))
        self.hcr.estimate_coefficients_nd(data)
        return self
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using the HCR regressor.

        Args:
            X (np.ndarray): Samples to predict.

        Returns:
            np.ndarray: Predicted values.
        """
        return np.array([self.hcr.expected_value(-1) for _ in range(len(X))])