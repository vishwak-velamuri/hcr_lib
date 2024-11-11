import numpy as np
import yfinance as yf
from scipy import stats, special
from datetime import datetime
from itertools import product, combinations
from functools import reduce

class RecurrentHCR:
    def __init__(self, dimension, degree):
        self.dimension = dimension
        self.degree = degree
        self.marginal_coefficients = []
        
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
        normalized = stats.t.cdf(log_returns, *params)
        normalized = np.clip(normalized, 1e-10, 1 - 1e-10)
        return normalized
    
    def create_n_tuples(self, normalized_data):
        """Create n-dimensional tuples from normalized data."""
        n = self.dimension
        return np.column_stack([normalized_data[i:-(n-i-1) if n-i-1 > 0 else None] 
                              for i in range(n)])
    
    def legendre_basis(self, x):
        """Generate Legendre basis functions evaluated at x."""
        x = np.clip(x, 1e-10, 1 - 1e-10)
        x_transformed = 2 * x - 1
        return np.array([np.sqrt(2 * i + 1) * special.eval_legendre(i, x_transformed) 
                        for i in range(self.degree + 1)])
    
    def evaluate_1d_polynomials(self, points):
        """Evaluate Legendre polynomials for 1D points."""
        points = np.clip(points, 1e-10, 1 - 1e-10)
        x_transformed = 2 * points - 1
        n_points = len(points)
        pval = np.zeros((self.degree + 1, n_points))
        
        for i in range(self.degree + 1):
            pval[i] = np.sqrt(2 * i + 1) * special.eval_legendre(i, x_transformed)
                
        return pval
    
    def compute_marginal_density(self, point, coefficients):
        """Compute marginal density at a point."""
        basis_values = self.legendre_basis(point)
        density = 1.0 + np.sum(coefficients[1:] * basis_values[1:])  # Start from index 1
        return max(density, 1e-10)
    
    def estimate_marginal_coefficients(self, points, dim_idx, weights=None):
        """Estimate coefficients for marginal distribution."""
        current_points = points[:, dim_idx]
        pval = self.evaluate_1d_polynomials(current_points)
        
        if weights is None:
            weights = np.ones(len(points)) / len(points)
        else:
            weights = weights / np.sum(weights)
            
        coefficients = np.zeros(self.degree + 1)
        coefficients[0] = 1.0  # Set first coefficient to 1
        for i in range(1, self.degree + 1):  # Start from index 1
            coefficients[i] = np.sum(weights * pval[i])
            
        return coefficients
    
    def compute_pairwise_interaction(self, points, dim1, dim2):
        """Compute pairwise interaction terms between two dimensions."""
        points1 = points[:, dim1]
        points2 = points[:, dim2]
        
        pval1 = self.evaluate_1d_polynomials(points1)
        pval2 = self.evaluate_1d_polynomials(points2)
        
        interaction_terms = np.zeros(len(points))
        
        for i in range(1, self.degree + 1):
            for j in range(1, self.degree + 1):
                coef = np.mean(pval1[i] * pval2[j])
                interaction_terms += coef * pval1[i] * pval2[j]
                
        return interaction_terms * 0.5  # Scale interaction terms
    
    def fit_recurrent(self):
        """Fit the recurrent HCR model using DJI data."""
        # Get and prepare data
        prices = self.get_data()
        log_returns = self.calculate_log_returns(prices)
        normalized_data = self.normalize_data(log_returns)
        points = self.create_n_tuples(normalized_data)
        
        self.marginal_coefficients = []
        density = np.ones(len(points))
        
        # Fit marginal distributions
        for dim in range(self.dimension):
            coeffs = self.estimate_marginal_coefficients(points, dim)
            self.marginal_coefficients.append(coeffs)
            marginal_density = np.array([self.compute_marginal_density(p, coeffs) 
                                       for p in points[:, dim]])
            density *= marginal_density
        
        # Add pairwise interactions
        if self.dimension > 1:
            for dim1, dim2 in combinations(range(self.dimension), 2):
                interaction = self.compute_pairwise_interaction(points, dim1, dim2)
                density *= (1.0 + interaction)
        
        # Calculate log-likelihood
        log_likelihood = np.mean(np.log(np.maximum(density, 1e-10)))
        
        return points, log_likelihood
    
    def compute_density(self, point):
        """Compute full density at a point."""
        if len(self.marginal_coefficients) == 0:
            raise ValueError("Model must be fitted before computing density")
            
        density = 1.0
        # Marginal terms
        for dim in range(self.dimension):
            marginal = self.compute_marginal_density(point[dim], 
                                                   self.marginal_coefficients[dim])
            density *= marginal
            
        # Pairwise interactions
        if self.dimension > 1:
            for dim1, dim2 in combinations(range(self.dimension), 2):
                interaction = 0.5 * np.sum([self.legendre_basis(point[dim1])[i] * 
                                          self.legendre_basis(point[dim2])[j]
                                          for i in range(1, self.degree + 1)
                                          for j in range(1, self.degree + 1)])
                density *= (1.0 + interaction)
                
        return density

def main():
    """Test RecurrentHCR with different dimensions."""
    dimensions = [1, 2, 3]
    degree = 6
    
    for dim in dimensions:
        print(f"\nTesting {dim}-dimensional HCR:")
        hcr = RecurrentHCR(dimension=dim, degree=degree)
        _, log_likelihood = hcr.fit_recurrent()
        print(f"Dimension: {dim}")
        print(f"Log-likelihood: {log_likelihood:.6f}")

if __name__ == "__main__":
    main()