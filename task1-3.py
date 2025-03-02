import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose, STL


# Read the second sheet of the Excel file (index starts from 0, so the second sheet has an index of 1)
file_path = './data/statistic_id1048518_global-land-temperature-anomalies-1880-2020.xlsx'
df = pd.read_excel(file_path, sheet_name=1)

# Extract the second column (time) and the third column (anomalies) data, ignoring empty cells
time_data = df.iloc[:, 1].dropna()  # Second column data
anomalies_data = df.iloc[:, 2].dropna()  # Third column data

# Filter out non-numeric data
time_data = time_data.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)
anomalies_data = anomalies_data.apply(pd.to_numeric, errors='coerce').dropna().reset_index(drop=True)

# Convert pandas Series to numpy arrays
time_data = time_data.to_numpy()
anomalies_data = anomalies_data.to_numpy()

# Take the first order difference
diff_anomalies_data = pd.Series(anomalies_data).diff().dropna().to_numpy()

####################Original dataset#############################
# Line plot
plt.figure(figsize=(12, 6))
plt.plot(time_data, anomalies_data)
plt.title('Global Land Temperature Anomalies Over Time')
plt.xlabel('Year')
plt.ylabel('Temperature Anomalies')
plt.savefig('pics/1-3-line_plot.png', dpi=500)
plt.show()

# Histogram
plt.figure(figsize=(12, 6))
plt.hist(anomalies_data, bins=30, density=True, alpha=0.6, color='g')
plt.title('Histogram of Temperature Anomalies')
plt.xlabel('Temperature Anomalies')
plt.ylabel('Frequency')
plt.savefig('pics/1-3-histogram.png', dpi=500)
plt.show()

# Density plot
plt.figure(figsize=(12, 6))
sns.kdeplot(anomalies_data, shade=True)
plt.title('Density Plot of Temperature Anomalies')
plt.xlabel('Temperature Anomalies')
plt.ylabel('Density')
plt.savefig('pics/1-3-density_plot.png', dpi=500)
plt.show()

# Box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x=anomalies_data)
plt.title('Box Plot of Temperature Anomalies')
plt.xlabel('Temperature Anomalies')
plt.savefig('pics/1-3-box_plot.png', dpi=500)
plt.show()

# Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(anomalies_data.reshape(-1, 1), cmap='coolwarm', cbar=True)
plt.title('Heatmap of Temperature Anomalies')
plt.xlabel('Temperature Anomalies')
plt.savefig('pics/1-3-heatmap.png', dpi=500)
plt.show()

# Lag-1 plot
plt.figure(figsize=(12, 6))
plt.scatter(anomalies_data[:-1], anomalies_data[1:])
plt.title('Lag-1 Plot of Temperature Anomalies')
plt.xlabel('Temperature Anomalies (t)')
plt.ylabel('Temperature Anomalies (t+1)')
plt.savefig('pics/1-3-lag1_plot.png', dpi=500)
plt.show()

# ACF plot
plt.figure(figsize=(12, 6))
plot_acf(anomalies_data, lags=40)
plt.title('ACF Plot of Temperature Anomalies')
plt.savefig('pics/1-3-acf_plot.png', dpi=500)
plt.show()

# PACF plot
plt.figure(figsize=(12, 6))
plot_pacf(anomalies_data, lags=40)
plt.title('PACF Plot of Temperature Anomalies')
plt.savefig('pics/1-3-pacf_plot.png', dpi=500)
plt.show()

########################First order differencing########################
# Line plot
plt.figure(figsize=(12, 6))
plt.plot(time_data[1:], diff_anomalies_data)
plt.title('First Order Difference of Global Land Temperature Anomalies Over Time')
plt.xlabel('Year')
plt.ylabel('First Order Difference of Temperature Anomalies')
plt.savefig('pics/1-3-diff_line_plot.png', dpi=500)
plt.show()

# Histogram
plt.figure(figsize=(12, 6))
plt.hist(diff_anomalies_data, bins=30, density=True, alpha=0.6, color='g')
plt.title('Histogram of First Order Difference of Temperature Anomalies')
plt.xlabel('First Order Difference of Temperature Anomalies')
plt.ylabel('Frequency')
plt.savefig('pics/1-3-diff_histogram.png', dpi=500)
plt.show()

# Density plot
plt.figure(figsize=(12, 6))
sns.kdeplot(diff_anomalies_data, shade=True)
plt.title('Density Plot of First Order Difference of Temperature Anomalies')
plt.xlabel('First Order Difference of Temperature Anomalies')
plt.ylabel('Density')
plt.savefig('pics/1-3-diff_density_plot.png', dpi=500)
plt.show()

# Box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x=diff_anomalies_data)
plt.title('Box Plot of First Order Difference of Temperature Anomalies')
plt.xlabel('First Order Difference of Temperature Anomalies')
plt.savefig('pics/1-3-diff_box_plot.png', dpi=500)
plt.show()

# Heatmap
plt.figure(figsize=(12, 6))
sns.heatmap(diff_anomalies_data.reshape(-1, 1), cmap='coolwarm', cbar=True)
plt.title('Heatmap of First Order Difference of Temperature Anomalies')
plt.xlabel('First Order Difference of Temperature Anomalies')
plt.savefig('pics/1-3-diff_heatmap.png', dpi=500)
plt.show()

# Lag-1 plot
plt.figure(figsize=(12, 6))
plt.scatter(diff_anomalies_data[:-1], diff_anomalies_data[1:])
plt.title('Lag-1 Plot of First Order Difference of Temperature Anomalies')
plt.xlabel('First Order Difference of Temperature Anomalies (t)')
plt.ylabel('First Order Difference of Temperature Anomalies (t+1)')
plt.savefig('pics/1-3-diff_lag1_plot.png', dpi=500)
plt.show()

# ACF plot
plt.figure(figsize=(12, 6))
plot_acf(diff_anomalies_data, lags=40)
plt.title('ACF Plot of First Order Difference of Temperature Anomalies')
plt.savefig('pics/1-3-diff_acf_plot.png', dpi=500)
plt.show()

# PACF plot
plt.figure(figsize=(12, 6))
plot_pacf(diff_anomalies_data, lags=40)
plt.title('PACF Plot of First Order Difference of Temperature Anomalies')
plt.savefig('pics/1-3-diff_pacf_plot.png', dpi=500)
plt.show()

########################Randomness Test########################
print("Ljung-Box test for original series:")
lb_test_original = acorr_ljungbox(anomalies_data, lags=[40], return_df=True)
print(lb_test_original)

print("\nLjung-Box test for differenced series:")
lb_test_diff = acorr_ljungbox(diff_anomalies_data, lags=[40], return_df=True)
print(lb_test_diff)

########################Stationarity tests using ADF test########################
print("\nADF test for original series:")
adf_test_original = adfuller(anomalies_data)
print(f"ADF Statistic: {adf_test_original[0]}")
print(f"p-value: {adf_test_original[1]}")
print("Critical Values:")
for key, value in adf_test_original[4].items():
    print(f"   {key}: {value}")

print("\nADF test for differenced series:")
adf_test_diff = adfuller(diff_anomalies_data)
print(f"ADF Statistic: {adf_test_diff[0]}")
print(f"p-value: {adf_test_diff[1]}")
print("Critical Values:")
for key, value in adf_test_diff[4].items():
    print(f"   {key}: {value}")

#######################Classical decomposition and STL decomposition#############
result_additive = seasonal_decompose(anomalies_data, model='additive', period=12)
result_additive.plot()
plt.savefig('pics/1-3-stl_decomposition.png', dpi=500)
# print(result_additive.trend)
# print(result_additive.seasonal)
# print(result_additive.resid)
plt.show()


# Convert anomalies_data to pandas Series
anomalies_series = pd.Series(anomalies_data)

stl = STL(anomalies_series, period=12)

result_stl = stl.fit()
result_stl.plot()
# print(result_stl.trend)
# print(result_stl.seasonal)
# print(result_stl.resid)
plt.show()
