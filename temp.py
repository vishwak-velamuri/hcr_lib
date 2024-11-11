import numpy as np
import yfinance as yf
from scipy import stats, special
from datetime import datetime
from itertools import product
from functools import reduce
from sklearn.model_selection import KFold

class HCR:
    def __init__(self, dimension, degree):
        """
        Initialize HCR with specified dimension and polynomial degree.
        
        Args:
            dimension (int): Number of dimensions for the reconstruction
            degree (int): Degree of Legendre polynomials
        """
        self.dimension = dimension
        self.degree = degree
    
    def get_data(self):
        """Get DJI financial data from Yahoo Finance."""
        start_date = datetime(2000, 1, 1)
        dji = yf.download('^DJI', start=start_date)
        return dji['Close'].values
    
    def calculate_log_returns(self, prices):
        """Calculate log returns from price data."""
        return np.log(prices[1:] / prices[:-1])
    
    def normalize_data(self, log_returns):
        """Normalize data using Student's t-distribution."""
        params = stats.t.fit(log_returns)
        return stats.t.cdf(log_returns, *params)
    
    def create_n_tuples(self, normalized_data):
        """Create n-dimensional tuples from normalized data."""
        n = self.dimension
        return np.column_stack([normalized_data[i:-(n-i-1) if n-i-1 > 0 else None] 
                              for i in range(n)])
    
    def legendre_basis(self, x):
        """
        Generate Legendre basis functions evaluated at x.
        
        Args:
            x (float): Point to evaluate basis functions at
        Returns:
            list: Values of basis functions at x
        """
        x_transformed = 2 * x - 1
        return [np.sqrt(2 * i + 1) * special.eval_legendre(i, x_transformed) 
                for i in range(self.degree + 1)]
    
    def evaluate_polynomials(self, n_tuples):
        """
        Evaluate Legendre polynomials for all dimensions and degrees.
        
        Args:
            n_tuples (numpy.ndarray): Array of n-dimensional points
        Returns:
            numpy.ndarray: Evaluated polynomials
        """
        n_points = len(n_tuples)
        pval = np.ones((self.degree + 1, self.dimension, n_points))
        
        for i in range(1, self.degree + 1):
            for j in range(self.dimension):
                pval[i, j, :] = np.sqrt(2 * i + 1) * special.eval_legendre(i, 2 * n_tuples[:, j] - 1)
                
        return pval
    
    def estimate_coefficients(self, pval):
        """
        Estimate coefficients for n-dimensional HCR.
        
        Args:
            pval (numpy.ndarray): Evaluated polynomials
        Returns:
            numpy.ndarray: Estimated coefficients
        """
        shape = [self.degree + 1] * self.dimension
        coefficients = np.zeros(shape)
        
        index_combinations = product(range(self.degree + 1), repeat=self.dimension)
        
        for indices in index_combinations:
            term = reduce(lambda x, y: x * y, 
                        [pval[idx, dim, :] for dim, idx in enumerate(indices)])
            coefficients[indices] = np.mean(term)
            
        return coefficients
    
    def compute_density(self, point, coefficients):
        """
        Compute density at a specific point.
        
        Args:
            point (numpy.ndarray): Point to evaluate density at
            coefficients (numpy.ndarray): HCR coefficients
        Returns:
            float: Density value at the point
        """
        basis_values = [self.legendre_basis(x) for x in point]
        
        result = 0
        for indices in product(range(self.degree + 1), repeat=self.dimension):
            coef = coefficients[indices]
            basis_term = reduce(lambda x, y: x * y, 
                              [basis_values[dim][idx] for dim, idx in enumerate(indices)])
            result += coef * basis_term
            
        return result

    def cross_validation(self, n_tuples, n_folds=10):
        """
        Perform 10-fold cross-validation and compute average log-likelihood.
        
        Args:
            n_tuples (numpy.ndarray): N-dimensional data points
            n_folds (int): Number of folds for cross-validation (default: 10)
        Returns:
            float: Average log-likelihood across all folds
        """
        # Initialize K-fold cross-validation
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
        fold_log_likelihoods = []
        
        # Perform k-fold cross-validation
        for fold, (train_idx, test_idx) in enumerate(kf.split(n_tuples), 1):
            # Split data into training and test sets
            train_data = n_tuples[train_idx]
            test_data = n_tuples[test_idx]
            
            # Train on the training set
            pval_train = self.evaluate_polynomials(train_data)
            coefficients = self.estimate_coefficients(pval_train)
            
            # Evaluate on the test set
            test_log_likelihoods = []
            for point in test_data:
                density = self.compute_density(point, coefficients)
                if density > 0:  # Avoid log(0)
                    test_log_likelihoods.append(np.log(density))
            
            # Calculate mean log-likelihood for this fold
            fold_ll = np.mean(test_log_likelihoods)
            fold_log_likelihoods.append(fold_ll)
        
        # Calculate and return the average log-likelihood across all folds
        mean_ll = np.mean(fold_log_likelihoods)
        std_ll = np.std(fold_log_likelihoods)
        return mean_ll, std_ll
    
    def fit(self):
        """
        Fit the HCR model to DJI data.
        
        Returns:
            tuple: (n_tuples, coefficients, (mean_log_likelihood, std_log_likelihood))
        """
        # Get and prepare data
        prices = self.get_data()
        log_returns = self.calculate_log_returns(prices)
        normalized_data = self.normalize_data(log_returns)
        n_tuples = self.create_n_tuples(normalized_data)
        
        # Fit model and perform cross-validation
        pval = self.evaluate_polynomials(n_tuples)
        coefficients = self.estimate_coefficients(pval)
        cv_results = self.cross_validation(n_tuples)
        
        return n_tuples, coefficients, cv_results

def main():
    # Test HCR with different dimensions
    dimensions = [1, 2, 3]
    degree = 6
    
    for dim in dimensions:
        print(f"\nTesting {dim}-dimensional HCR:")
        hcr = HCR(dimension=dim, degree=degree)
        _, _, (mean_ll, std_ll) = hcr.fit()
        print(f"Final {dim}D Results - Log-likelihood: {mean_ll:.6f} ± {std_ll:.6f}")

if __name__ == "__main__":
    main()