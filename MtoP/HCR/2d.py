import numpy as np
import yfinance as yf
from scipy import stats, special
from datetime import datetime
import matplotlib.pyplot as plt

m = 6 # Degree of the Legendre polynomial basis

def get_financial_data():
    start_date = datetime(2000, 1, 1)
    dji = yf.download('^DJI', start=start_date)
    return dji['Close'].values # Return closing prices

def calculate_log_returns(prices):
    return np.log(prices[1:] / prices[:-1]) # Calculate log returns between successive price points

def normalize_data(log_returns):
    # Normalize the log returns using the Student's t-distribution
    params = stats.t.fit(log_returns) # Fit t-distribution parameters to log returns
    return stats.t.cdf(log_returns, *params) # Normalize log returns using CDF of the fitted t-distribution

def create_pairs(normalized_data):
    return np.column_stack((normalized_data[:-1], normalized_data[1:])) # Stack columns of consecutive pairs of consecutive normalized data points (x(t), x(t+1))

def legendre_basis(m):
    def basis_function(x, n):
        # Generate the nth Legendre polynomial evaluated at a transformed x
        x_transformed = 2 * x - 1 # Transform x to range [-1, 1]
        return np.sqrt(2 * n + 1) * special.eval_legendre(n, x_transformed) # Evaluate nth Legendre polynomial
    
    return [lambda x, n=n: basis_function(x, n) for n in range(m + 1)] # Return a list of basis functions for each degree up to m

def evaluate_polynomials(pairs, basis_functions):
    n_pairs = len(pairs)
    pval = np.ones((m + 1, 2, n_pairs))

    for i in range(1, m + 1):
        pval[i, 0, :] = basis_functions[i](pairs[:, 0]) # Evaluate polynomial for x(t)
        pval[i, 1, :] = basis_functions[i](pairs[:, 1]) # Evaluate polynomial for x(t+1)

    return pval

def estimate_coefficients(pval):
    a = np.zeros((m + 1, m + 1)) # Initialize coefficient matrix

    for i in range(m + 1):
        for j in range(m + 1):
            a[i, j] = np.mean(pval[i, 0, :] * pval[j, 1, :]) # Compute coefficients using cross-terms

    return a

def plot_density_and_points(pairs, coefficients, basis_functions):
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, 1, 100)
    X, Y = np.meshgrid(x, y)
    Z = np.zeros_like(X)

    for i in range(m + 1):
        for j in range(m + 1):
            Z += coefficients[i, j] * basis_functions[i](X) * basis_functions[j](Y)
    
    plt.figure(figsize=(10, 8))
    contour = plt.contourf(X, Y, Z, levels=20, cmap='plasma')
    plt.colorbar(contour, label='Density')
    plt.scatter(pairs[:, 0], pairs[:, 1], c='black', alpha=0.1, s=10)
    plt.xlabel('x(t)')
    plt.ylabel('x(t+1)')
    plt.title('Joint Distribution of Normalized Log Returns')
    plt.show()

def main():
    dat = get_financial_data()
    lr = calculate_log_returns(dat)
    ndat = normalize_data(lr)
    pairs = create_pairs(ndat)
    basis_functions = legendre_basis(m)
    pval = evaluate_polynomials(pairs, basis_functions)
    a = estimate_coefficients(pval)
    plot_density_and_points(pairs, a, basis_functions)
    return dat, lr, ndat, pairs, pval, a

def cross_validation(pairs, basis_functions, m):
    np.random.shuffle(pairs)
    train, test = np.array_split(pairs, 2) # Split data into training and testing sets
    pval = np.ones((m + 1, 2, len(train))) # Initialize array to store evaluated polynomials for training data

    for i, b in enumerate(basis_functions):
        pval[i, 0, :] = b(train[:, 0])  # Evaluate basis functions on training x(t)
        pval[i, 1, :] = b(train[:, 1])  # Evaluate basis functions on training x(t+1)

    # Estimate coefficients based on training data
    a = np.zeros((m + 1, m + 1))

    for i in range(m + 1):
        for j in range(m + 1):
            a[i, j] = np.mean(pval[i, 0, :] * pval[j, 1, :])

    # Create lattice for the density grid
    nn = 100
    lat = (np.arange(nn) + 0.5) / nn  # Create lattice points in range (0, 1)
    pbas = np.array([b(lat) for b in basis_functions])  # Evaluate basis functions on the lattice
    rho = pbas.T @ a @ pbas  # Compute density on the lattice
    rho = np.maximum(rho, 0.3)  # Calibrate the density to enforce minimum value
    rho *= nn**2 / np.sum(rho)  # Normalize the density to ensure it sums to 1

    # Evaluate the model's log-likelihood on the test set
    test_indices = np.clip((nn * test).astype(int), 0, nn-1)  # Convert test points to indices on the lattice
    ll = np.mean(np.log([rho[v[0], v[1]] for v in test_indices]))  # Compute mean log-likelihood
    print(f"HCR degree m={m}, log-likelihood = {ll:.7f}")

if __name__ == "__main__":
    dat, lr, ndat, pairs, pval, a = main()
    basis_functions = legendre_basis(m)
    cross_validation(pairs, basis_functions, m)