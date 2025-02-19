import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller, acf, pacf, q_stat
from scipy.stats import boxcox
from statsmodels.stats.diagnostic import acorr_ljungbox

# Parameters
N = 1000  # You can change this to 100, 5000, or 10000

# Generate random walk series
np.random.seed(42)
steps = np.random.choice([-1, 1], size=N)
random_walk = np.cumsum(steps)
random_walk = np.insert(random_walk, 0, 0)  # Start from y0 = 0

# Calculate statistics
mean = np.mean(random_walk)
std_dev = np.std(random_walk)

print(f"Mean: {mean}")
print(f"Standard Deviation: {std_dev}")

# Plotting
plt.figure(figsize=(14, 12))

# Line plot
plt.subplot(3, 2, 1)
plt.plot(random_walk)
plt.title('Random Walk Line Plot')

# Histogram
plt.subplot(3, 2, 2)
plt.hist(random_walk, bins=30, density=True, alpha=0.6, color='g')
plt.title('Histogram')

# Density plot
plt.subplot(3, 2, 3)
density, bins, _ = plt.hist(random_walk, bins=30, density=True, alpha=0.0)
count, _ = np.histogram(random_walk, bins)
plt.plot(bins[1:], density)
plt.title('Density Plot')

# Box plot
plt.subplot(3, 2, 4)
plt.boxplot(random_walk, vert=False)
plt.title('Box Plot')

# Lag-1 plot
plt.subplot(3, 2, 5)
plt.scatter(random_walk[:-1], random_walk[1:])
plt.title('Lag-1 Plot')

# ACF and PACF plots
plt.figure(figsize=(12, 8))
plt.subplot(2, 1, 1)
plot_acf(random_walk, lags=40, ax=plt.gca())
plt.title('ACF Plot')

plt.subplot(2, 1, 2)
plot_pacf(random_walk, lags=40, ax=plt.gca())
plt.title('PACF Plot')

plt.tight_layout()
plt.show()

# Ljung-Box test
lb_test = acorr_ljungbox(random_walk, lags=[40], return_df=True)
print("Ljung-Box test results:")
print(lb_test)

# Augmented Dickey-Fuller test
adf_test = adfuller(random_walk)
print("ADF test results:")
print(f"ADF Statistic: {adf_test[0]}")
print(f"p-value: {adf_test[1]}")
print("Critical Values:")
for key, value in adf_test[4].items():
    print(f"   {key}: {value}")