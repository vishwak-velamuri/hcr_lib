import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy import stats, special
import yfinance as yf
from datetime import datetime

# Set the degree of the polynomial
m = 2

def get_financial_data():
    start_date = datetime(2000, 1, 1)
    dji = yf.download('^DJI', start=start_date)
    return dji['Close'].values

def calculate_log_returns(prices):
    return np.log(prices[1:] / prices[:-1])

def normalize_data(log_returns):
    # Normalize the log returns using the Student's t-distribution
    params = stats.t.fit(log_returns)
    return stats.t.cdf(log_returns, *params)

# Define the Legendre basis functions
def legendre_basis(x, m):
    return [np.sqrt(2 * i + 1) * special.eval_legendre(i, 2 * x - 1) for i in range(m + 1)]

dat = get_financial_data()
lr = calculate_log_returns(dat)
ndat = normalize_data(lr)

# Create triples from the normalized data
triples = np.column_stack((ndat[:-2], ndat[1:-1], ndat[2:]))

# Evaluate basis functions for the triples
pval = np.ones((m + 1, 3, len(triples)))
for i in range(1, m + 1):
    for j in range(3):
        pval[i, j, :] = legendre_basis(triples[:, j], i)[-1]

# Estimate coefficients
a = np.zeros((m + 1, m + 1, m + 1))
for i in range(m + 1):
    for j in range(m + 1):
        for k in range(m + 1):
            a[i, j, k] = np.mean(pval[i, 0, :] * pval[j, 1, :] * pval[k, 2, :])

def rho(x, y, z):
    result = 0
    basis_x = legendre_basis(x, m)
    basis_y = legendre_basis(y, m)
    basis_z = legendre_basis(z, m)
    for i in range(m + 1):
        for j in range(m + 1):
            for k in range(m + 1):
                result += a[i, j, k] * basis_x[i] * basis_y[j] * basis_z[k]
    return result

fig = plt.figure(figsize=(12, 10))
ax = fig.add_subplot(111, projection='3d')
x = y = z = np.linspace(0, 1, 20)
X, Y, Z = np.meshgrid(x, y, z)
values = np.vectorize(rho)(X, Y, Z)
scatters = ax.scatter(X, Y, Z, c=values, cmap='plasma', alpha=0.1)

plt.colorbar(scatters, label='Density')
ax.scatter(triples[:, 0], triples[:, 1], triples[:, 2], color='red', alpha=0.5, s=1)
ax.set_xlabel('x(t)')
ax.set_ylabel('x(t+1)')
ax.set_zlabel('x(t+2)')
ax.set_title('3D Density Visualization of Normalized Log Returns')
plt.show()