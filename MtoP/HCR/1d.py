import numpy as np
from scipy import special, integrate
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Parameters
pts = 100  # Number of generated points (samples)
mm = 5     # Maximum polynomial degree for density estimation

# Define the true density function ρ(x)
# This represents our assumed density function on the interval [0,1]
def rho(x):
    return 2 * x**3 + 6 * (x - 0.5)**2  # ρ(x) must be nonnegative and integrate to 1

# Generate sample points using the CDF
def generate_samples(pts):
    x_vals = np.linspace(0, 1, 1001)  # Create an array of values from 0 to 1
    # Calculate the CDF using numerical integration of the density function
    # CDF(x) = ∫ρ(t)dt from [0,x]
    cdf_vals = [integrate.quad(rho, 0, y)[0] for y in x_vals]
    # Interpolate the CDF to find the inverse CDF (for sampling)
    inverse_cdf = interp1d(cdf_vals, x_vals)
    # Generate uniform random samples between 0 and 1
    uniform_samples = np.random.uniform(0, 1, pts)
    # Transform the uniform samples to the distribution defined by the CDF
    return inverse_cdf(uniform_samples)

# Generate orthonormal basis using Legendre polynomials
def legendre_basis(x, m):
    # Transform x from [0,1] to [-1,1] for Legendre polynomial evaluation
    x_transformed = 2 * x - 1
    # Calculate the m-th Legendre polynomial and normalize
    return special.eval_legendre(m, x_transformed) * np.sqrt(2 * m + 1)

# Main computation
dt = generate_samples(pts)  # Generate sample points from the true density

# Calculate coefficients a_m, which are the averages of the basis functions over the samples
a = np.zeros(mm)  # Array to store the coefficients
x_continuous = np.linspace(0, 1, 1000)  # Create a continuous range for plotting
for m in range(mm):
    # Calculate the mean of the m-th Legendre polynomial evaluated at the sample points
    a[m] = np.mean([legendre_basis(x, m) for x in dt])

# Calculate estimated densities based on the coefficients and the basis functions
rhos = [rho(x_continuous)]  # Start with the true density
for m in range(1, mm + 1):
    density = np.ones_like(x_continuous)  # Initialize density array
    # Combine contributions from the basis functions weighted by their coefficients
    for j in range(m):
        density += a[j] * legendre_basis(x_continuous, j)
    rhos.append(density)  # Append the estimated density for this degree

# Plotting
plt.figure(figsize=(12, 8))
# Plot the true density and estimated densities separately
for i, density in enumerate(rhos):
    label = "real ρ" if i == 0 else f"Degree {i}"
    plt.plot(x_continuous, density, label=label)

# Plot sample points as vertical lines
for point in dt:
    plt.plot([point, point], [0, 0.5], color='orange', alpha=0.5)  # Sample points in orange

plt.legend()
plt.xlabel('x')
plt.ylabel('Density')
plt.title('Density Estimation using Legendre Polynomials')
plt.show()