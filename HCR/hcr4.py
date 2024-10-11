import numpy as np
from scipy import stats
from scipy.special import legendre
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

class HCR:
    def __init__(self, max_degree=5):
        self.max_degree = max_degree
        self._prepare_basis()
    
    def _prepare_basis(self):
        """Prepare normalized Legendre polynomials."""
        self.basis_funcs = []
        for m in range(self.max_degree + 1):
            norm_factor = np.sqrt(2 * m + 1)
            leg_poly = legendre(m)
            
            def func(x, m=m, norm_factor=norm_factor, leg_poly=leg_poly):
                x_transformed = 2 * x - 1
                return norm_factor * leg_poly(x_transformed)
            
            self.basis_funcs.append(func)
    
    @staticmethod
    def norm_gaussian(data):
        """Normalize data using Gaussian CDF."""
        mean, std = np.mean(data), np.std(data)
        return stats.norm.cdf((data - mean) / std)
    
    @staticmethod
    def norm_edf(data):
        """Normalize data using Empirical Distribution Function."""
        n = len(data)
        ranks = stats.rankdata(data, method='average')
        return (ranks - 1) / (n - 1) if n > 1 else 0.5
    
    def evaluate_basis(self, x, degrees=None):
        """Evaluate basis functions at given points."""
        if degrees is None:
            degrees = range(self.max_degree + 1)
        
        result = np.zeros((len(degrees), len(x)))
        for i, m in enumerate(degrees):
            result[i] = self.basis_funcs[m](x)
        return result
    
    def estimate_coefficients_1d(self, data):
        """Estimate coefficients for 1D density."""
        basis_values = self.evaluate_basis(data)
        return np.mean(basis_values, axis=1)
    
    def estimate_coefficients_2d(self, data):
        """Estimate coefficients for 2D density."""
        x, y = data[:, 0], data[:, 1]
        basis_x = self.evaluate_basis(x)
        basis_y = self.evaluate_basis(y)
        
        coeffs = np.zeros((self.max_degree + 1, self.max_degree + 1))
        for i in range(self.max_degree + 1):
            for j in range(self.max_degree + 1):
                coeffs[i, j] = np.mean(basis_x[i] * basis_y[j])
        return coeffs
    
    def estimate_coefficients_3d(self, data):
        """Estimate coefficients for 3D density."""
        x, y, z = data[:, 0], data[:, 1], data[:, 2]
        basis_x = self.evaluate_basis(x)
        basis_y = self.evaluate_basis(y)
        basis_z = self.evaluate_basis(z)
        
        coeffs = np.zeros((self.max_degree + 1, self.max_degree + 1, self.max_degree + 1))
        for i in range(self.max_degree + 1):
            for j in range(self.max_degree + 1):
                for k in range(self.max_degree + 1):
                    coeffs[i, j, k] = np.mean(basis_x[i] * basis_y[j] * basis_z[k])
        return coeffs

    def density_1d(self, coeffs, x):
        """Evaluate 1D density at given points."""
        basis_values = self.evaluate_basis(x)
        return np.dot(coeffs, basis_values)
    
    def density_2d(self, coeffs, x, y):
        """Evaluate 2D density at given points."""
        basis_x = self.evaluate_basis(x)
        basis_y = self.evaluate_basis(y)
        
        result = np.zeros((len(x), len(y)))
        for i in range(self.max_degree + 1):
            for j in range(self.max_degree + 1):
                result += coeffs[i, j] * np.outer(basis_x[i], basis_y[j])
        return result
    
    def calibrate_density(self, density, min_value=0.1):
        """Calibrate density to ensure non-negative values and proper normalization."""
        calibrated = np.maximum(density, min_value)
        
        if calibrated.ndim == 1:
            dx = 1.0 / len(calibrated)
            integral = np.sum(calibrated) * dx
            calibrated /= integral
        elif calibrated.ndim == 2:
            dx = 1.0 / calibrated.shape[0]
            dy = 1.0 / calibrated.shape[1]
            integral = np.sum(calibrated) * dx * dy
            calibrated /= integral
        
        return calibrated
    
    def log_likelihood(self, coeffs, data, dimension=1):
        """Calculate log-likelihood of the data given the model.
        
        Args:
            coeffs: Model coefficients
            data: Data points (normalized)
            dimension: 1, 2, or 3 for different dimensional models
        
        Returns:
            Average log-likelihood per point
        """
        if dimension == 1:
            density_values = self.density_1d(coeffs, data)
        elif dimension == 2:
            x, y = data[:, 0], data[:, 1]
            density_values = np.array([
                self.density_2d(coeffs, np.array([xi]), np.array([yi]))[0, 0]
                for xi, yi in zip(x, y)
            ])
        else:
            raise ValueError("Only 1D and 2D implemented for now")
        
        # Calibrate density values
        density_values = np.maximum(density_values, 0.1)  # Minimum density as in article
        
        # Calculate log-likelihood
        return np.mean(np.log(density_values))
    
    def cross_validate(self, data, k_folds=5, dimension=1):
        """Perform k-fold cross-validation and return log-likelihoods.
        
        Args:
            data: Input data (normalized)
            k_folds: Number of folds for cross-validation
            dimension: 1 or 2 for different dimensional models
        
        Returns:
            List of log-likelihoods for each degree
        """
        n_samples = len(data)
        fold_size = n_samples // k_folds
        indices = np.random.permutation(n_samples)
        
        log_likelihoods = []
        
        for degree in range(self.max_degree + 1):
            fold_lls = []
            
            for fold in range(k_folds):
                # Split data into train and test
                test_indices = indices[fold * fold_size:(fold + 1) * fold_size]
                train_indices = np.concatenate([
                    indices[:fold * fold_size],
                    indices[(fold + 1) * fold_size:]
                ])
                
                train_data = data[train_indices]
                test_data = data[test_indices]
                
                # Estimate coefficients on training data
                if dimension == 1:
                    coeffs = self.estimate_coefficients_1d(train_data)
                elif dimension == 2:
                    coeffs = self.estimate_coefficients_2d(train_data)
                
                # Calculate log-likelihood on test data
                ll = self.log_likelihood(coeffs, test_data, dimension)
                fold_lls.append(ll)
            
            avg_ll = np.mean(fold_lls)
            log_likelihoods.append(avg_ll)
        
        return log_likelihoods

def example_1d():
    """Reproduce the 1D example from the article."""
    # Define the true density function (2x^3 + 6(x-1/2)^2)
    def true_density(x):
        return 2 * x**3 + 6 * (x - 0.5)**2

    # Generate sample points
    np.random.seed(42)
    n_points = 100
    
    # Create sample points using rejection sampling
    x_samples = []
    max_density = true_density(1)  # approximate maximum
    while len(x_samples) < n_points:
        x = np.random.uniform(0, 1)
        y = np.random.uniform(0, max_density)
        if y <= true_density(x):
            x_samples.append(x)
    x_samples = np.array(x_samples)
    
    # Create HCR instance and estimate density
    hcr = HCR(max_degree=5)
    coeffs = hcr.estimate_coefficients_1d(x_samples)
    
    # Evaluate estimated density
    x_eval = np.linspace(0, 1, 1000)
    estimated_density = hcr.density_1d(coeffs, x_eval)
    calibrated_density = hcr.calibrate_density(estimated_density)
    
    # Plotting
    plt.figure(figsize=(12, 6))
    plt.plot(x_eval, true_density(x_eval), 'b-', label='True Density')
    plt.plot(x_eval, calibrated_density, 'r--', label='Estimated Density')
    plt.hist(x_samples, bins=20, density=True, alpha=0.3, label='Sample Histogram')
    plt.xlabel('x')
    plt.ylabel('Density')
    plt.legend()
    plt.title('HCR 1D Density Estimation')
    plt.show()
    
    return coeffs, x_eval, calibrated_density

def example_2d():
    """Reproduce a 2D example similar to the financial data in the article."""
    # Generate correlated data
    np.random.seed(42)
    n_points = 1000
    
    # Generate X and Y with some dependency
    x = np.random.normal(0, 1, n_points)
    y = 0.7 * x + 0.3 * np.random.normal(0, 1, n_points)
    
    # Normalize data
    hcr = HCR(max_degree=6)
    x_norm = hcr.norm_gaussian(x)
    y_norm = hcr.norm_gaussian(y)
    data_2d = np.column_stack([x_norm, y_norm])
    
    # Estimate coefficients and density
    coeffs_2d = hcr.estimate_coefficients_2d(data_2d)
    
    # Create grid for evaluation
    x_grid = np.linspace(0, 1, 50)
    y_grid = np.linspace(0, 1, 50)
    X, Y = np.meshgrid(x_grid, y_grid)
    
    Z = hcr.density_2d(coeffs_2d, x_grid, y_grid)
    Z = hcr.calibrate_density(Z)
    
    # Plotting
    fig = plt.figure(figsize=(12, 5))
    
    ax1 = fig.add_subplot(121)
    ax1.scatter(x_norm, y_norm, alpha=0.1)
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Normalized Data')
    
    ax2 = fig.add_subplot(122)
    c = ax2.contourf(X, Y, Z, levels=20)
    plt.colorbar(c)
    ax2.set_xlabel('X')
    ax2.set_ylabel('Y')
    ax2.set_title('Estimated 2D Density')
    
    plt.tight_layout()
    plt.show()
    
    return coeffs_2d, Z

def example_cross_validation():
    """Example demonstrating cross-validation on different datasets."""
    np.random.seed(42)
    hcr = HCR(max_degree=6)
    
    # 1D Example
    print("1D Cross-validation example:")
    # Generate data from a mixture of Gaussians
    n_samples = 1000
    data_1d = np.concatenate([
        np.random.normal(0, 1, n_samples // 2),
        np.random.normal(3, 0.5, n_samples // 2)
    ])
    
    # Normalize data
    data_norm = hcr.norm_gaussian(data_1d)
    
    # Perform cross-validation
    lls_1d = hcr.cross_validate(data_norm, k_folds=5, dimension=1)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(lls_1d)), lls_1d, '-o')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Average Log-Likelihood')
    plt.title('1D Cross-Validation Results')
    plt.grid(True)
    plt.show()
    
    # 2D Example
    print("\n2D Cross-validation example:")
    # Generate correlated 2D data
    x = np.random.normal(0, 1, n_samples)
    y = 0.7 * x + 0.3 * np.random.normal(0, 1, n_samples)
    
    # Normalize data
    x_norm = hcr.norm_gaussian(x)
    y_norm = hcr.norm_gaussian(y)
    data_2d = np.column_stack([x_norm, y_norm])
    
    # Perform cross-validation
    lls_2d = hcr.cross_validate(data_2d, k_folds=5, dimension=2)
    
    # Plot results
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(lls_2d)), lls_2d, '-o')
    plt.xlabel('Polynomial Degree')
    plt.ylabel('Average Log-Likelihood')
    plt.title('2D Cross-Validation Results')
    plt.grid(True)
    plt.show()
    
    return lls_1d, lls_2d

def run_tests():
    """Run basic tests to verify the implementation."""
    hcr = HCR(max_degree=3)
    
    # Test 1: Normalization
    data = np.random.normal(0, 1, 1000)
    norm_g = hcr.norm_gaussian(data)
    norm_e = hcr.norm_edf(data)
    assert np.all((norm_g >= 0) & (norm_g <= 1)), "Gaussian normalization failed"
    assert np.all((norm_e >= 0) & (norm_e <= 1)), "EDF normalization failed"
    
    # Test 2: Basis functions
    x = np.linspace(0, 1, 100)
    basis_vals = hcr.evaluate_basis(x)
    assert basis_vals.shape == (4, 100), "Basis evaluation shape incorrect"
    
    # Test 3: Coefficients estimation and density evaluation
    coeffs_1d = hcr.estimate_coefficients_1d(norm_g)
    density_1d = hcr.density_1d(coeffs_1d, x)
    calibrated_1d = hcr.calibrate_density(density_1d)
    assert np.all(calibrated_1d >= 0), "Calibrated density contains negative values"
    
    # Test 4: 2D estimation
    data_2d = np.column_stack([norm_g, norm_e])
    coeffs_2d = hcr.estimate_coefficients_2d(data_2d)
    assert coeffs_2d.shape == (4, 4), "2D coefficients shape incorrect"
    
    print("All tests passed successfully!")

if __name__ == "__main__":
    # Run tests
    run_tests()
    
    # Run examples
    print("\nRunning 1D example...")
    coeffs_1d, x_eval_1d, density_1d = example_1d()
    
    print("\nRunning 2D example...")
    coeffs_2d, density_2d = example_2d()

    # Run cross-validation examples
    print("Running cross-validation examples...")
    lls_1d, lls_2d = example_cross_validation()