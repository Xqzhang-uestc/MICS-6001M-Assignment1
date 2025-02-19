from random import gauss, seed
from pandas import Series
import matplotlib.pyplot as plt
import pandas as pd
from pandas.plotting import autocorrelation_plot
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller

# Set the random seed for reproducibility
seed(10)

# Generate white noise series
N = 1000  # Number of data points
series = [gauss(0.0, 1.0) for i in range(N)]
series = Series(series)

# Print summary statistics
print("Summary Statistics:")
print(series.describe())

# print mean and standard deviation
mean = series.mean()
std_dev = series.std()
print(f"\nActual Mean: {mean}")
print(f"Actual Standard Deviation: {std_dev}")

# 1. Line Chart
plt.figure(figsize=(12, 6))
plt.plot(series, label="White Noise Series")
plt.title("Line Plot of White Noise Series")
plt.xlabel("Time")
plt.ylabel("Value")
plt.legend()
plt.show()

# 2. Histogram
plt.figure(figsize=(12, 6))
series.hist(bins=30, density=True, alpha=0.6, color='g', label='Histogram')
plt.title("Histogram of White Noise Series")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.legend()
plt.show()

# 3. Density Plot
plt.figure(figsize=(12, 6))
series.plot(kind='kde', color='r', label='Density Plot')
plt.title("Density Plot of White Noise Series")
plt.xlabel("Value")
plt.ylabel("Density")
plt.legend()
plt.show()

# 4. Box Plot
plt.figure(figsize=(8, 6))
series.plot(kind='box', vert=False)
plt.title("Box Plot of White Noise Series")
plt.show()

# 5. Lag-1 Plot
plt.figure(figsize=(8, 6))
pd.plotting.lag_plot(series, lag=1)
plt.title("Lag-1 Plot of White Noise Series")
plt.xlabel("Series(t)")
plt.ylabel("Series(t+1)")
plt.show()

# 6. ACF Plot
plt.figure(figsize=(12, 6))
autocorrelation_plot(series)
plt.title("ACF Plot of White Noise Series")
plt.show()

# 7. ACF and PACF Plots (up to 40 lags)
plt.figure(figsize=(12, 6))
plot_acf(series, lags=40)
plt.title("ACF Plot (up to 40 lags)")
plt.show()

plt.figure(figsize=(12, 6))
plot_pacf(series, lags=40)
plt.title("PACF Plot (up to 40 lags)")
plt.show()

# Generate 100 random series
num_series = 100
length = 1000
random_series_list = [[gauss(0.0, 1.0) for _ in range(length)] for _ in range(num_series)]
random_series_list = [Series(series) for series in random_series_list]

# Compute the average series
average_series = pd.concat(random_series_list, axis=1).mean(axis=1)
print("\nSummary Statistics of Average Series:")
print(average_series.describe())

# Perform Ljung-Box test
ljung_box_test = acorr_ljungbox(series, lags=[40], return_df=True)
print("\nLjung-Box Test Results:")
print(ljung_box_test)

# Perform Augmented Dickey-Fuller test
adf_result = adfuller(series)
print("\nADF Test Results:")
print(f"ADF Statistic: {adf_result[0]}")
print(f"p-value: {adf_result[1]}")
print(f"Critical Values: {adf_result[4]}")