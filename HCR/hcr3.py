import numpy as np
from scipy import stats, special
import matplotlib.pyplot as plt

class HCR:
    def __init__(self, max_degree=6, normalization='edf', calibration=True, adaptation_rate=0.95, lattice_size=100):
        self.max_degree = max_degree
        self.normalization = normalization
        self.calibration = calibration
        self.adaptation_rate = adaptation_rate
        self.lattice_size = lattice_size
        self.basis = None
        self.coefficients = None
        self.lattice = None
        self.normalization_constant = None
        self.dim = None

    def normalize(self, data):
        if self.normalization == 'gaussian':
            return stats.norm.cdf((data - np.mean(data)) / np.std(data))
        elif self.normalization == 'student_t':
            params = stats.t.fit(data)
            return stats.t.cdf(data, *params)
        elif self.normalization == 'edf':
            return stats.rankdata(data) / (len(data) + 1)
        else:
            raise ValueError("Unknown normalization method")

    def prepare_basis(self):
        self.basis = [lambda t, i=i: special.eval_legendre(i, 2*t - 1) * np.sqrt(2*i + 1) for i in range(self.max_degree + 1)]
        self.lattice = np.linspace(0, 1, self.lattice_size)

    def estimate_coefficients(self, data):
        self.dim = data.shape[1]
        self.prepare_basis()
        
        if self.dim == 1:
            coefficients = np.array([np.mean([f(x) for x in data]) for f in self.basis[1:]])
        elif self.dim == 2:
            coefficients = np.array([[np.mean([f1(x)*f2(y) for x, y in data]) for f2 in self.basis[1:]] for f1 in self.basis[1:]])
        elif self.dim == 3:
            coefficients = np.array([[[np.mean([f1(x)*f2(y)*f3(z) for x, y, z in data]) for f3 in self.basis[1:]] for f2 in self.basis[1:]] for f1 in self.basis[1:]])
        
        self.coefficients = coefficients
        self.calculate_normalization_constant()
        return coefficients

    def calculate_normalization_constant(self):
        if self.dim == 1:
            density_values = np.array([self.uncalibrated_density(x) for x in self.lattice])
        elif self.dim == 2:
            density_values = np.array([[self.uncalibrated_density([x, y]) for x in self.lattice] for y in self.lattice])
        elif self.dim == 3:
            density_values = np.array([[[self.uncalibrated_density([x, y, z]) for x in self.lattice] for y in self.lattice] for z in self.lattice])
        
        if self.calibration:
            density_values = np.maximum(density_values, 0.1)
        
        self.normalization_constant = np.sum(density_values) * (1 / self.lattice_size) ** self.dim

    def uncalibrated_density(self, x):
        rho = 1
        if self.dim == 1:
            x = [x] if np.isscalar(x) else x
            rho += np.sum([a * f(x[0]) for a, f in zip(self.coefficients, self.basis[1:])])
        elif self.dim == 2:
            rho += np.sum([a * f1(x[0]) * f2(x[1]) for a, f1 in zip(self.coefficients.flat, self.basis[1:]) for f2 in self.basis[1:]])
        elif self.dim == 3:
            rho += np.sum([a * f1(x[0]) * f2(x[1]) * f3(x[2]) for a, f1 in zip(self.coefficients.flat, self.basis[1:]) for f2 in self.basis[1:] for f3 in self.basis[1:]])
        return rho

    def density(self, x):
        rho = self.uncalibrated_density(x)
        if self.calibration:
            rho = max(rho, 0.1)
        return rho / self.normalization_constant

    def log_likelihood(self, data):
        return np.mean(np.log([max(self.density(x), 1e-10) for x in data]))

    def fit(self, data):
        normalized_data = np.array([self.normalize(data[:,i]) for i in range(data.shape[1])]).T
        self.estimate_coefficients(normalized_data)

    def update(self, new_data):
        normalized_new_data = np.array([self.normalize(new_data[:,i]) for i in range(new_data.shape[1])]).T
        new_coefficients = self.estimate_coefficients(normalized_new_data)
        self.coefficients = self.adaptation_rate * self.coefficients + (1 - self.adaptation_rate) * new_coefficients
        self.calculate_normalization_constant()

# Example usage and testing
import yfinance as yf
from sklearn.model_selection import train_test_split

# Download Dow Jones Industrial Average data
djia = yf.Ticker("^DJI").history(period="max")

# Calculate log returns
log_returns = np.log(djia['Close'] / djia['Close'].shift(1)).dropna().values

# Split data into training and testing sets
train_data, test_data = train_test_split(log_returns, test_size=0.2, random_state=42)

# Create and fit HCR model
hcr = HCR(max_degree=6, normalization='student_t', calibration=True)
hcr.fit(train_data.reshape(-1, 1))

# Calculate log-likelihood for training and testing data
train_ll = hcr.log_likelihood(train_data.reshape(-1, 1))
test_ll = hcr.log_likelihood(test_data.reshape(-1, 1))
print(f"1D Log-likelihood - Train: {train_ll:.4f}, Test: {test_ll:.4f}")

# Plot 1D density
x = np.linspace(0, 1, 1000)
y = [hcr.density([xi]) for xi in x]
plt.figure(figsize=(10, 6))
plt.plot(x, y)
plt.title("1D HCR Density Estimation")
plt.xlabel("Normalized Log Returns")
plt.ylabel("Density")
plt.savefig("hcr_1d_density.png")
plt.close()

# 2D example with pairs of succeeding log returns
pairs = np.array([log_returns[:-1], log_returns[1:]]).T
train_pairs, test_pairs = train_test_split(pairs, test_size=0.2, random_state=42)

hcr_2d = HCR(max_degree=6, normalization='student_t', calibration=True)
hcr_2d.fit(train_pairs)

# Calculate 2D log-likelihood
train_ll_2d = hcr_2d.log_likelihood(train_pairs)
test_ll_2d = hcr_2d.log_likelihood(test_pairs)
print(f"2D Log-likelihood - Train: {train_ll_2d:.4f}, Test: {test_ll_2d:.4f}")

# Plot 2D density
x = y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)
Z = np.array([[hcr_2d.density([xi, yi]) for xi in x] for yi in y])

plt.figure(figsize=(10, 8))
plt.contourf(X, Y, Z, levels=20, cmap='viridis')
plt.colorbar(label='Density')
plt.title("2D HCR Density Estimation")
plt.xlabel("Normalized Log Returns (t)")
plt.ylabel("Normalized Log Returns (t+1)")
plt.savefig("hcr_2d_density.png")
plt.close()

# 3D example with triples of succeeding log returns
triples = np.array([log_returns[:-2], log_returns[1:-1], log_returns[2:]]).T
train_triples, test_triples = train_test_split(triples, test_size=0.2, random_state=42)

hcr_3d = HCR(max_degree=2, normalization='student_t', calibration=True)
hcr_3d.fit(train_triples)

# Calculate 3D log-likelihood
train_ll_3d = hcr_3d.log_likelihood(train_triples)
test_ll_3d = hcr_3d.log_likelihood(test_triples)
print(f"3D Log-likelihood - Train: {train_ll_3d:.4f}, Test: {test_ll_3d:.4f}")

# Experiment with different polynomial degrees
degrees = range(2, 9)
train_lls = []
test_lls = []

for degree in degrees:
    hcr = HCR(max_degree=degree, normalization='student_t', calibration=True)
    hcr.fit(train_data.reshape(-1, 1))
    train_lls.append(hcr.log_likelihood(train_data.reshape(-1, 1)))
    test_lls.append(hcr.log_likelihood(test_data.reshape(-1, 1)))

plt.figure(figsize=(10, 6))
plt.plot(degrees, train_lls, label='Train')
plt.plot(degrees, test_lls, label='Test')
plt.title("Log-likelihood vs Polynomial Degree")
plt.xlabel("Polynomial Degree")
plt.ylabel("Log-likelihood")
plt.legend()
plt.savefig("hcr_degree_comparison.png")
plt.close()

# Experiment with different adaptation rates
adaptation_rates = np.linspace(0.8, 0.99, 10)
test_lls = []

for rate in adaptation_rates:
    hcr = HCR(max_degree=6, normalization='student_t', calibration=True, adaptation_rate=rate)
    hcr.fit(train_data.reshape(-1, 1))
    
    # Simulate online learning
    for i in range(0, len(test_data), 100):
        hcr.update(test_data[i:i+100].reshape(-1, 1))
    
    test_lls.append(hcr.log_likelihood(test_data.reshape(-1, 1)))

plt.figure(figsize=(10, 6))
plt.plot(adaptation_rates, test_lls)
plt.title("Log-likelihood vs Adaptation Rate")
plt.xlabel("Adaptation Rate")
plt.ylabel("Log-likelihood")
plt.savefig("hcr_adaptation_rate_comparison.png")
plt.close()

print("HCR analysis complete. Check the generated plots for visualizations.")