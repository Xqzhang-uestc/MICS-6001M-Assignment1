import numpy as np
import matplotlib.pyplot as plt

# Time Series
x = np.array([1, 2, 3, 4, 5, 6, 7, 8, 7, 6, 5, 4, 3, 2, 1])
N = len(x)

# Calculate the mean
mu = np.mean(x)

# Calculate autocovariance
def autocovariance(x, k, mu):
    N = len(x)
    sum_val = 0
    for t in range(N - k):
        sum_val += (x[t] - mu) * (x[t+k] - mu)
    return sum_val / N

# Calculate autocorrelation
def autocorrelation(x, k, mu):
    return autocovariance(x, k, mu) / autocovariance(x, 0, mu)

lags = range(N)
autocov = [autocovariance(x, k, mu) for k in lags]
autocorr = [autocorrelation(x, k, mu) for k in lags]

print("Autocovariances:", autocov)
print("Autocorrelations:", autocorr)


plt.figure(figsize=(10, 6))
plt.stem(lags, autocorr)
plt.title("ACF Graph")
plt.savefig('pics/2-3-discussion-ACF Graph.png', dpi=500)
plt.xlabel("Lag")
plt.ylabel("Autocorrelation")
plt.xlim([-1, N])
plt.grid(True)
plt.show()
