import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import yfinance as yf
from datetime import datetime

def get_financial_data():
    start_date = datetime(2000, 1, 1)
    dji = yf.download('^DJI', start=start_date)
    return dji['Close'].values

def calculate_log_returns(prices):
    return np.log(prices[1:] / prices[:-1])

def normalize_data(log_returns):
    params = stats.t.fit(log_returns)
    return stats.t.cdf(log_returns, *params)

def create_pairs(normalized_data):
    return np.column_stack((normalized_data[:-1], normalized_data[1:]))

# Get and process data
dat = get_financial_data()
lr = calculate_log_returns(dat)
ndat = normalize_data(lr)
pairs = create_pairs(ndat)

# Split data into train and test sets
np.random.shuffle(pairs)
split_index = len(pairs) // 2
train, test = pairs[:split_index], pairs[split_index:]

# Perform KDE
kde = stats.gaussian_kde(train.T)

# Calculate log-likelihood
log_likelihood = np.mean(np.log(kde(test.T)))
print(f"log-likelihood = {log_likelihood:.6f}")

# Create a grid for contour plot
x, y = np.mgrid[0:1:100j, 0:1:100j]
positions = np.vstack([x.ravel(), y.ravel()])
z = np.reshape(kde(positions), x.shape)

# Create the plot
fig, ax = plt.subplots(figsize=(10, 8))

# Contour plot
contour = ax.contourf(x, y, z, levels=20, cmap='viridis')
plt.colorbar(contour, label='Density')

# Scatter plot of original pairs
ax.scatter(pairs[:, 0], pairs[:, 1], c='red', alpha=0.1, s=1)

ax.set_xlabel('x(t)')
ax.set_ylabel('x(t+1)')
ax.set_title('KDE of Normalized Log Returns')

plt.show()