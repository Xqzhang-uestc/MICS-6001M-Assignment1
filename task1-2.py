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



# Line plot
plt.figure(figsize=(12, 6))
plt.plot(random_walk)
plt.title('Random Walk Line Plot')
plt.savefig('pics/1-2-line_plot.png', dpi=500)

# Histogram
plt.figure(figsize=(12, 6))
plt.hist(random_walk, bins=30, density=True, alpha=0.6, color='g')
plt.title('Histogram')
plt.savefig('pics/1-2-histogram.png', dpi=500)

# Density plot
plt.figure(figsize=(12, 6))
density, bins, _ = plt.hist(random_walk, bins=30, density=True, alpha=0.0)
count, _ = np.histogram(random_walk, bins)
plt.plot(bins[1:], density)
plt.title('Density Plot')
plt.savefig('pics/1-2-density_plot.png', dpi=500)

# Box plot
plt.figure(figsize=(12, 6))
plt.boxplot(random_walk, vert=False)
plt.title('Box Plot')
plt.savefig('pics/1-2-box_plot.png', dpi=500)

# Lag-1 plot
plt.figure(figsize=(12, 6))
plt.scatter(random_walk[:-1], random_walk[1:])
plt.title('Lag-1 Plot')
plt.savefig('pics/1-2-lag1_plot.png', dpi=500)


# ACF and PACF plots
plt.figure(figsize=(12, 6))
plot_acf(random_walk, lags=40, ax=plt.gca())
plt.title('ACF Plot')
plt.savefig('pics/1-2-acf_plot.png', dpi=500)

plt.figure(figsize=(12, 6))
plot_pacf(random_walk, lags=40, ax=plt.gca())
plt.title('PACF Plot')
plt.savefig('pics/1-2-pacf_plot.png', dpi=500)



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