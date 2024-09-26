import numpy as np
from scipy import stats
from scipy.special import legendre
from itertools import product
from typing import Callable, List, Tuple
from sklearn.model_selection import KFold
from sklearn.linear_model import LinearRegression
from sklearn.decomposition import PCA

class HCR:
    def __init__(self, m: int, data: np.ndarray = None, time_dependent: bool = False, alpha: float = 0.1):
        """
        Initialize the HCR (Hierarchical Correlation Reconstruction) object.

        Args:
            m (int): Maximum degree of Legendre polynomials.
            data (np.ndarray, optional): Data for determining dimensionality (d). Defaults to None.
            time_dependent (bool): Whether to use time-dependent modeling. Defaults to False.
            alpha (float): Smoothing factor for time-dependent modeling. Defaults to 0.1.
        """
        self.m = m
        if data is not None:
            if len(data.shape) != 2:
                raise ValueError("Data should be a 2D array (samples, features).")
            self.d = data.shape[1]
        else:
            self.d = 1
        self.coefficients = np.zeros((self.m + 1) ** self.d)
        self.time_dependent = time_dependent
        self.alpha = alpha

    @staticmethod
    def normalize_gaussian(data: np.ndarray) -> np.ndarray:
        """Normalize data to ~uniform[0,1] using Gaussian CDF."""
        mean = np.mean(data)
        std = np.std(data)
        if std == 0:
            raise ValueError("Standard deviation of the data is zero. Cannot normalize.")
        return stats.norm.cdf((data - mean) / std)

    @staticmethod
    def normalize_edf(data: np.ndarray) -> np.ndarray:
        """Normalize data to ~uniform[0,1] using Empirical Distribution Function."""
        sorted_data = np.sort(data)
        ranks = np.argsort(data)
        n = len(data)
        if n == 0:
            raise ValueError("Input data is empty.")
        edf = (ranks + 0.5) / n
        return edf

    @staticmethod
    def legendre_basis(m: int, x: np.ndarray) -> np.ndarray:
        """Generate orthonormal Legendre polynomial basis up to degree m."""
        if len(x) == 0:
            raise ValueError("Input array x is empty.")
        basis = np.zeros((m + 1, len(x)))
        for i in range(m + 1):
            basis[i] = np.sqrt(2 * i + 1) * legendre(i)(x)
        return basis

    def estimate_coefficients(self, data: np.ndarray) -> np.ndarray:
        """Estimate coefficients for n-dimensional data."""
        if data.shape[1] != self.d:
            raise ValueError(f"Input data has incorrect dimensions. Expected {self.d}, got {data.shape[1]}.")
        
        index_combinations = list(product(range(self.m + 1), repeat=self.d))
        self.coefficients = np.zeros(len(index_combinations))

        for idx, combination in enumerate(index_combinations):
            basis_product = np.ones(data.shape[0])
            for dim, degree in enumerate(combination):
                basis_i = self.legendre_basis(degree, data[:, dim])
                basis_product *= basis_i[degree, :]
            
            if self.time_dependent:
                self.coefficients[idx] = self._exponential_moving_average(basis_product)
            else:
                self.coefficients[idx] = np.mean(basis_product)

        return self.coefficients

    def _exponential_moving_average(self, data: np.ndarray) -> float:
        """Calculate exponential moving average for time-dependent modeling."""
        ema = data[0]
        for i in range(1, len(data)):
            ema = self.alpha * data[i] + (1 - self.alpha) * ema
        return ema

    def density(self, x: np.ndarray) -> float:
        """Calculate density for n-dimensional points."""
        if len(x.shape) != 2 or x.shape[1] != self.d:
            raise ValueError(f"Input points have incorrect dimensions. Expected {self.d}-dimensional data.")
        basis_functions = [self.legendre_basis(self.m, x[:, i]) for i in range(self.d)]
        basis_product = np.array([np.prod(comb) for comb in product(*basis_functions)])
        densities = np.dot(self.coefficients, basis_product.T)
        return self._calibrate_density(densities)

    def _calibrate_density(self, density: float, min_density: float = 1e-6) -> float:
        """Calibrate density to ensure non-negativity."""
        return np.maximum(density, min_density)

    def conditional_density(self, x: np.ndarray, given: List[int]) -> np.ndarray:
        """Calculate conditional density given specific dimensions."""
        full_density = self.density(x)
        marginal_density = self.marginal_density(x, given)
        return full_density / marginal_density

    def marginal_density(self, x: np.ndarray, dims: List[int]) -> np.ndarray:
        """Calculate marginal density for specified dimensions."""
        marginal_coeffs = self._marginalize_coefficients(dims)
        basis_functions = [self.legendre_basis(self.m, x[:, i]) for i in dims]
        basis_product = np.array([np.prod(comb) for comb in product(*basis_functions)])
        return np.dot(marginal_coeffs, basis_product)

    def _marginalize_coefficients(self, dims: List[int]) -> np.ndarray:
        """Marginalize coefficients for specified dimensions."""
        index_combinations = list(product(range(self.m + 1), repeat=len(dims)))
        marginal_coeffs = np.zeros(len(index_combinations))
        for idx, combination in enumerate(index_combinations):
            full_combination = [0] * self.d
            for i, dim in enumerate(dims):
                full_combination[dim] = combination[i]
            marginal_coeffs[idx] = self.coefficients[self._combination_to_index(full_combination)]
        return marginal_coeffs

    def _combination_to_index(self, combination: Tuple[int, ...]) -> int:
        """Convert a combination of indices to a flat index."""
        return sum(i * (self.m + 1) ** j for j, i in enumerate(combination))

    def cross_validate(self, data: np.ndarray, k: int = 5) -> float:
        """Perform k-fold cross-validation and return mean log-likelihood."""
        kf = KFold(n_splits=k, shuffle=True, random_state=42)
        log_likelihoods = []

        for train_index, test_index in kf.split(data):
            train_data, test_data = data[train_index], data[test_index]
            self.estimate_coefficients(train_data)
            log_likelihoods.append(np.mean(np.log(self.density(test_data))))

        return np.mean(log_likelihoods)

    def predict_conditional_distribution(self, x: np.ndarray, y: np.ndarray) -> Callable:
        """Predict conditional distribution of y given x using linear regression."""
        reg = LinearRegression().fit(x, y)
        y_pred = reg.predict(x)
        residuals = y - y_pred
        
        # Estimate density of residuals
        residual_hcr = HCR(self.m, data=residuals.reshape(-1, 1))
        residual_hcr.estimate_coefficients(residuals.reshape(-1, 1))
        
        def conditional_distribution(x_new):
            y_pred_new = reg.predict(x_new.reshape(1, -1))
            return lambda y: residual_hcr.density((y - y_pred_new).reshape(1, -1))
        
        return conditional_distribution

    def analyze_correlations(self, data: np.ndarray, n_components: int = 2) -> Tuple[np.ndarray, np.ndarray]:
        """Perform correlation analysis using PCA."""
        pca = PCA(n_components=n_components)
        transformed_data = pca.fit_transform(data)
        
        correlations = np.zeros((n_components, self.d))
        for i in range(n_components):
            for j in range(self.d):
                correlations[i, j] = np.corrcoef(transformed_data[:, i], data[:, j])[0, 1]
        
        return pca.components_, correlations

    def calculate_conditional_entropy(self, data: np.ndarray, target_dim: int) -> Tuple[float, float]:
        """Calculate conditional entropy for a specific dimension."""
        full_entropy = self._calculate_entropy(data)
    
        # Create a new HCR instance for the marginal data
        marginal_data = np.delete(data, target_dim, axis=1)
        marginal_hcr = HCR(self.m, data=marginal_data)
        marginal_hcr.estimate_coefficients(marginal_data)
        marginal_entropy = marginal_hcr._calculate_entropy(marginal_data)
    
        relevance = float(max(full_entropy - marginal_entropy, 0))  # Already float
    
        # Create a new HCR instance for the target dimension data
        target_data = data[:, target_dim].reshape(-1, 1)
        target_hcr = HCR(self.m, data=target_data)
        target_hcr.estimate_coefficients(target_data)
        novelty = float(max(target_hcr._calculate_entropy(target_data), 0))  # Ensure float
    
        return relevance, novelty

    def _calculate_entropy(self, data: np.ndarray) -> float:
        """Calculate entropy of the given data."""
        # Create a new HCR instance if the data dimensionality doesn't match
        if data.shape[1] != self.d:
            temp_hcr = HCR(self.m, data=data)
            temp_hcr.estimate_coefficients(data)
            densities = temp_hcr.density(data)
        else:
            self.estimate_coefficients(data)
            densities = self.density(data)
        
        # Ensure densities are positive and not too close to zero
        densities = np.maximum(densities, 1e-10)
        
        calculated_entropy = -np.mean(np.log(densities))
        return max(0, calculated_entropy)