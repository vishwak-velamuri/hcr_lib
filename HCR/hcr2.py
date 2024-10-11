import numpy as np
from scipy import special, stats, linalg
from sklearn.decomposition import PCA
from sklearn.linear_model import Lasso, LinearRegression
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go
import yfinance as yf
import warnings

"""HCR (Hierarchical correlation reconstruction) Python Library"""

class HCR:
    def __init__(self, max_degree=6, dimensions=1, adaptation_rate = 0.95):
        self.max_degree = max_degree
        self.dimensions = dimensions
        self.adaptation_rate = adaptation_rate
        
    @staticmethod
    def legendre_polynomial(m, x):
        """Generate scaled Legendre polynomial of degree m"""
        x_scaled = 2 * x - 1
        scaling_factor = np.sqrt(2 * m + 1)
        return special.eval_legendre(m, x_scaled) * scaling_factor

    def generate_product_basis(self, points):
        """
        Generate product basis for multiple variables
        
        Parameters:
        points : array-like, shape (n_samples, n_dimensions)
        
        Returns:
        List of basis function evaluations
        """
        n_samples = points.shape[0]
        basis_values = np.ones((self.max_degree + 1, n_samples, self.dimensions))
        
        for d in range(self.dimensions):
            for m in range(self.max_degree + 1):
                basis_values[m, :, d] = self.legendre_polynomial(m, points[:, d])
        
        return basis_values

    def normalize_data(self, data, method='student'):
        """
        Normalize data to uniform distribution on [0,1] using various methods
        
        Parameters:
        data : array-like
        method : str, 'student' or 'empirical'
        
        Returns:
        normalized data
        """
        if method == 'student':
            params = stats.t.fit(data)
            return stats.t.cdf(data, *params)
        elif method == 'empirical':
            return stats.rankdata(data) / (len(data) + 1)
        elif method == 'gaussian':
            return stats.norm.cdf(data, data.mean(), data.std())
        else:
            raise ValueError(f"Unknown normalization method: {method}")

    def estimate_coefficients(self, normalized_data):
        """
        Estimate coefficients for multivariate case
        
        Parameters:
        normalized_data : array-like, shape (n_samples, n_dimensions)
        
        Returns:
        coefficient tensor
        """
        basis_values = self.generate_product_basis(normalized_data)
        
        # Initialize coefficient tensor
        coef_shape = [self.max_degree + 1] * self.dimensions
        coefficients = np.zeros(coef_shape)
        
        # Calculate coefficients
        for idx in np.ndindex(*coef_shape):
            product = np.ones(normalized_data.shape[0])
            for d, i in enumerate(idx):
                product *= basis_values[i, :, d]
            coefficients[idx] = np.mean(product)
            
        return coefficients

    def estimate_density(self, points, coefficients):
        """
        Estimate density at given points using coefficients
        
        Parameters:
        points : array-like, shape (n_points, n_dimensions)
        coefficients : array-like, coefficient tensor
        
        Returns:
        estimated density values
        """
        basis_values = self.generate_product_basis(points)
        density = np.zeros(points.shape[0])
        
        for idx in np.ndindex(*coefficients.shape):
            product = np.ones(points.shape[0])
            for d, i in enumerate(idx):
                product *= basis_values[i, :, d]
            density += coefficients[idx] * product
            
        return density
    
    def calibrate_density(self, density_values, min_value=0.3):
        """Calibrate density to ensure minimum value and proper normalization"""
        calibrated = np.maximum(density_values, min_value)
        # Ensure it still integrates to 1
        return calibrated / np.mean(calibrated)
    
    def cross_validate(self, data, k_folds=2):
        """
        Perform k-fold cross-validation
        
        Parameters:
        data : array-like, shape (n_samples, n_dimensions)
        k_folds : int, number of folds for cross-validation
        
        Returns:
        mean log-likelihood across folds
        """
        np.random.shuffle(data)
        fold_size = len(data) // k_folds
        log_likelihoods = []

        for i in range(k_folds):
            test_idx = slice(i * fold_size, (i + 1) * fold_size)
            test_data = data[test_idx]
            train_data = np.concatenate([data[:i * fold_size], data[(i + 1) * fold_size:]])
            
            # Train on training data
            coeffs = self.estimate_coefficients(train_data)
            
            # Evaluate on test data
            density_values = self.estimate_density(test_data, coeffs)
            calibrated_density = self.calibrate_density(density_values)
            
            # Calculate log-likelihood
            log_likelihoods.append(np.mean(np.log(calibrated_density)))

        return np.mean(log_likelihoods)

    def evaluate_kde(self, train_data, test_data):
        """Evaluate KDE performance for comparison"""
        kde = stats.gaussian_kde(train_data.T)
        log_likelihood = np.mean(np.log(kde(test_data.T)))
        return log_likelihood, kde
    
    def predict_conditional_distribution(self, X, Y, prediction_method='linear'):
        """
        Directly predict conditional probability distribution
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
        Y : array-like, shape (n_samples,)
        prediction_method : str, 'linear' or 'lasso'
        
        Returns:
        function that predicts conditional distribution coefficients
        """
        basis_y = self.generate_product_basis(Y.reshape(-1, 1))
        if prediction_method == 'linear':
            model = LinearRegression()
        elif prediction_method == 'lasso':
            model = Lasso(alpha=0.01)
        
        # Fit separate model for each coefficient
        coefficient_predictors = []
        for i in range(self.max_degree + 1):
            y = basis_y[i, :, 0]
            model.fit(X, y)
            coefficient_predictors.append(model)
        
        def predict_coefficients(X_new):
            return np.array([model.predict(X_new) for model in coefficient_predictors])
        
        return predict_coefficients

    def nonstationary_analysis(self, time_series, eta=0.1):
        """
        Perform non-stationary analysis with exponential moving average
        
        Parameters:
        time_series : array-like, shape (n_samples,)
        eta : float, learning rate
        
        Returns:
        coefficient_evolution : array-like, shape (n_times, max_degree + 1)
        """
        n_samples = len(time_series)
        coefficient_evolution = np.zeros((n_samples, self.max_degree + 1))
        
        # Initialize coefficients
        basis = self.generate_product_basis(time_series[0].reshape(1, -1))
        current_coeffs = np.mean(basis, axis=1)
        coefficient_evolution[0] = current_coeffs.flatten()
        
        for t in range(1, n_samples):
            x_t = time_series[t]
            basis_t = self.generate_product_basis(x_t.reshape(1, -1))
            
            # Update coefficients
            for j in range(self.max_degree + 1):
                current_coeffs[j] += eta * (basis_t[j, 0, 0] - current_coeffs[j])
            
            coefficient_evolution[t] = current_coeffs.flatten()
        
        return coefficient_evolution

    def multi_feature_correlation(self, X, Y, lags=None):
        """
        Perform multi-feature correlation analysis
        
        Parameters:
        X, Y : array-like, shape (n_samples,)
        lags : list of int, time lags to consider
        
        Returns:
        dominant_features : dict with PCA results for each lag
        """
        if lags is None:
            lags = [0]
        
        results = {}
        for lag in lags:
            if lag > 0:
                x_lagged = X[:-lag]
                y_lagged = Y[lag:]
            else:
                x_lagged = X
                y_lagged = Y
            
            # Generate joint basis
            joint_data = np.column_stack((x_lagged, y_lagged))
            joint_coeffs = self.estimate_coefficients(joint_data)
            
            # Flatten and prepare for PCA
            flat_coeffs = joint_coeffs.reshape(-1)
            
            # Perform PCA
            pca = PCA()
            pca.fit(flat_coeffs.reshape(1, -1))
            
            results[lag] = {
                'coefficients': joint_coeffs,
                'pca': pca
            }
        
        return results

    def canonical_correlation_analysis(self, X, Y):
        """
        Perform canonical correlation analysis
        
        Parameters:
        X : array-like, shape (n_samples, n_features_x)
        Y : array-like, shape (n_samples, n_features_y)
        
        Returns:
        correlations : array
        x_weights : array
        y_weights : array
        """
        # Center the data
        X -= X.mean(axis=0)
        Y -= Y.mean(axis=0)
        
        # Calculate covariance matrices
        C_xx = np.dot(X.T, X) / (X.shape[0] - 1)
        C_yy = np.dot(Y.T, Y) / (Y.shape[0] - 1)
        C_xy = np.dot(X.T, Y) / (X.shape[0] - 1)
        
        # Regularization for numerical stability
        reg = 1e-8
        C_xx += reg * np.eye(C_xx.shape[0])
        C_yy += reg * np.eye(C_yy.shape[0])
        
        # Solve eigenvalue problem
        inv_C_xx = linalg.inv(linalg.sqrtm(C_xx))
        inv_C_yy = linalg.inv(linalg.sqrtm(C_yy))
        M = np.dot(np.dot(inv_C_xx, C_xy), inv_C_yy)
        
        U, s, Vh = linalg.svd(M)
        
        x_weights = np.dot(inv_C_xx, U)
        y_weights = np.dot(inv_C_yy, Vh.T)
        
        return s, x_weights, y_weights

    def feature_evaluation(self, X, Y):
        """
        Evaluate features for relevance and novelty
        
        Parameters:
        X : array-like, shape (n_samples, n_features)
        Y : array-like, shape (n_samples,)
        
        Returns:
        relevance : array
        novelty : array
        """
        n_features = X.shape[1]
        relevance = np.zeros(n_features)
        novelty = np.zeros(n_features)
        
        # Calculate relevance (conditional entropy)
        for j in range(n_features):
            bins = min(int(np.sqrt(len(Y))), 50)  # adaptive binning
            joint_hist, _, _ = np.histogram2d(X[:, j], Y, bins=bins)

            # Normalize the joint histogram
            p_xy = joint_hist / float(np.sum(joint_hist))
            p_x = np.sum(p_xy, axis=1)  # marginal for x
            p_y = np.sum(p_xy, axis=0)  # marginal for y

            # Calculate mutual information
            mutual_info = 0.0
            for i in range(len(p_x)):
                for k in range(len(p_y)):
                    if p_xy[i, k] > 0:  # Avoid log(0)
                        mutual_info += p_xy[i, k] * np.log(p_xy[i, k] / (p_x[i] * p_y[k]))
        
            relevance[j] = mutual_info

        # Calculate novelty (information loss when removing feature)
        base_predictor = self.predict_conditional_distribution(X, Y)
        base_ll = self._calculate_log_likelihood(base_predictor, X, Y)
        
        for j in range(n_features):
            X_without_j = np.delete(X, j, axis=1)
            predictor_without_j = self.predict_conditional_distribution(X_without_j, Y)
            ll_without_j = self._calculate_log_likelihood(predictor_without_j, X_without_j, Y)
            novelty[j] = base_ll - ll_without_j
        
        return relevance, novelty

    def _calculate_log_likelihood(self, predictor, X, Y):
        """Helper method to calculate log-likelihood"""
        predicted_coeffs = predictor(X)
        Y_reshaped = Y.reshape(-1, 1) if len(Y.shape) == 1 else Y
        density_values = np.zeros(len(Y))

        for i in range(len(Y)):
            coeffs_for_point = [coef[i] for coef in predicted_coeffs]
            density_values[i] = self.estimate_density(Y_reshaped[i:i+1], np.array(coeffs_for_point))
    
        return np.mean(np.log(self.calibrate_density(density_values)))

def plot_3d_density(hcr, triples, coefficients):
    """Plot 3D density estimation with scatter plot using plotly"""
    x = y = z = np.linspace(0, 1, 20)
    X, Y, Z = np.meshgrid(x, y, z)
    points = np.column_stack((X.ravel(), Y.ravel(), Z.ravel()))
    
    density_values = hcr.estimate_density(points, coefficients)
    calibrated_density = hcr.calibrate_density(density_values)
    
    # Create plotly figure
    fig = go.Figure(data=[
        go.Volume(
            x=X.flatten(), y=Y.flatten(), z=Z.flatten(),
            value=calibrated_density,
            opacity=0.1,
            colorscale='Viridis',
        ),
        go.Scatter3d(
            x=triples[:, 0], y=triples[:, 1], z=triples[:, 2],
            mode='markers',
            marker=dict(size=2, color='black', opacity=0.5),
        )
    ])
    
    fig.update_layout(
        title='3D HCR Density Estimation',
        scene=dict(
            xaxis_title='x',
            yaxis_title='y',
            zaxis_title='z'
        )
    )
    
    fig.show()

def advanced_financial_analysis(returns, lags=[1, 5, 10, 15, 20]):
    hcr = HCR(max_degree=4)
    
    # Nonstationary analysis
    coeff_evolution = hcr.nonstationary_analysis(returns)
    
    # Multi-feature correlation
    corr_results = hcr.multi_feature_correlation(returns[:-1], returns[1:], lags=lags)
    
    # Prepare features for CCA
    feature_matrix = np.zeros((len(returns) - max(lags), len(lags)))
    for i, lag in enumerate(lags):
        feature_matrix[:, i] = returns[max(lags)-lag:-lag]
    target = returns[max(lags):]
    
    # Canonical correlation analysis
    correlations, x_weights, y_weights = hcr.canonical_correlation_analysis(
        feature_matrix, target.reshape(-1, 1))
    
    # Feature evaluation
    relevance, novelty = hcr.feature_evaluation(feature_matrix, target)
    
    return {
        'coefficient_evolution': coeff_evolution,
        'correlation_results': corr_results,
        'canonical_correlations': correlations,
        'feature_relevance': relevance,
        'feature_novelty': novelty
    }

def process_financial_data():
    """Process financial data similar to the Mathematica example"""
    # Download Dow Jones data
    dji = yf.download('^DJI', start='2000-01-01')['Adj Close']
    
    # Calculate log returns
    log_returns = np.log(dji / dji.shift(1)).dropna()
    
    # Initialize HCR for 2D case
    hcr_2d = HCR(max_degree=6, dimensions=2)
    hcr_3d = HCR(max_degree=2, dimensions=3)

    # Normalize log returns for both 2D and 3D analysis
    normalized_returns = hcr_2d.normalize_data(log_returns.values, method='student')
    
    # Create pairs of successive normalized returns
    pairs = np.column_stack((normalized_returns[:-1], normalized_returns[1:]))

    # Normalize log returns
    ll_hcr = hcr_2d.cross_validate(pairs)
    train_data, test_data = np.array_split(pairs, 2)
    ll_kde, kde = hcr_2d.evaluate_kde(train_data, test_data)
    
    print(f"HCR log-likelihood: {ll_hcr}")
    print(f"KDE log-likelihood: {ll_kde}")
    
    # Process triples for 3D analysis
    triples = np.column_stack((normalized_returns[:-2], normalized_returns[1:-1], normalized_returns[2:]))
    
    # Estimate coefficients for 3D case
    coefficients_3d = hcr_3d.estimate_coefficients(triples)
    
    # Plot 3D density
    plot_3d_density(hcr_3d, triples, coefficients_3d)
    
    return hcr_2d, hcr_3d, pairs, triples, normalized_returns

if __name__ == "__main__":
    warnings.filterwarnings('ignore')
    
    print("Starting financial data analysis...")
    
    # Run basic analysis
    hcr_2d, hcr_3d, pairs, triples, normalized_returns = process_financial_data()
    
    print("\nPerforming advanced financial analysis...")
    # Run advanced analysis with normalized returns
    advanced_results = advanced_financial_analysis(normalized_returns)
    
    # Print some results
    print("\nAdvanced Analysis Results:")
    print(f"Number of coefficients evolved: {advanced_results['coefficient_evolution'].shape[1]}")
    print(f"Number of time lags analyzed: {len(advanced_results['correlation_results'])}")
    print(f"Top canonical correlation: {advanced_results['canonical_correlations'][0]:.4f}")
    
    # Plot coefficient evolution
    plt.figure(figsize=(12, 6))
    for i in range(advanced_results['coefficient_evolution'].shape[1]):
        plt.plot(advanced_results['coefficient_evolution'][:, i], label=f'Coefficient {i}')
    plt.title('Coefficient Evolution Over Time')
    plt.legend()
    plt.show()
    
    print("\nAnalysis complete!")