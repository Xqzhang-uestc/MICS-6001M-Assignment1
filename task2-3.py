from scipy.io import loadmat                    # To load .mat files
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import numpy as np
from numpy import where
from numpy.fft import fft, rfft
from scipy.signal import spectrogram
# Load data
data = loadmat('./data/03_EEG-1-2seconds-1000Hz.mat')

# Extract time and EEG data
t = data['t'][0]
EEG = data['EEG'][:, 0]
dt = t[1] - t[0]  # Define the sampling interval
N = EEG.shape[0]    # Define the total number of data points
T = N * dt        # Define the total duration of the data
# Convert EEG data to pandas Series
eeg_series = pd.Series(EEG)

# Line plot
plt.figure(figsize=(12, 6))
plt.plot(t, EEG)
plt.title('EEG Signal Over Time')
plt.savefig('pics/2-3-Line plot-1000Hz.png', dpi=500)
plt.xlabel('Time [s]')
plt.ylabel(r'Voltage [$\mu$ V]')
plt.autoscale(tight=True)
# plt.show()

# Histogram
plt.figure(figsize=(12, 6))
plt.hist(eeg_series, bins=30, density=True, alpha=0.6, color='g')
plt.title('Histogram of EEG Signal')
plt.savefig('pics/2-3-Histogram-1000Hz.png', dpi=500)
plt.xlabel(r'Voltage [$\mu$ V]')
plt.ylabel('Frequency')
# plt.show()

# Density plot
plt.figure(figsize=(12, 6))
sns.kdeplot(eeg_series, shade=True)
plt.title('Density Plot of EEG Signal')
plt.savefig('pics/2-3-Density Plot-1000Hz.png', dpi=500)
plt.xlabel(r'Voltage [$\mu$ V]')
plt.ylabel('Density')
# plt.show()

# Box plot
plt.figure(figsize=(12, 6))
sns.boxplot(x=eeg_series)
plt.title('Box Plot of EEG Signal')
plt.savefig('pics/2-3-Box Plot-1000Hz.png', dpi=500)
plt.xlabel(r'Voltage [$\mu$ V]')
# plt.show()

# Lag-1 plot
plt.figure(figsize=(12, 6))
plt.scatter(eeg_series[:-1], eeg_series[1:])
plt.title('Lag-1 Plot of EEG Signal')
plt.savefig('pics/2-3-Lag-1 Plot-1000Hz.png', dpi=500)
plt.xlabel('EEG Signal (t)')
plt.ylabel('EEG Signal (t+1)')
# plt.show()

# ACF plot
plt.figure(figsize=(12, 6))
plot_acf(eeg_series, lags=50)
plt.title('ACF Plot of EEG Signal')
plt.savefig('pics/2-3-ACF Plot-1000Hz.png', dpi=500)
# plt.show()

# PACF plot
plt.figure(figsize=(12, 6))
plot_pacf(eeg_series, lags=50)
plt.title('PACF Plot of EEG Signal')
plt.savefig('pics/2-3-PACF Plot-1000Hz.png', dpi=500)
# plt.show()


mn = EEG.mean()  # Compute the mean of the data
vr = EEG.var()   # Compute the variance of the data
sd = EEG.std()   # Compute the standard deviation of the data

print('mn = ' + str(mn))
print('vr = ' + str(vr))
print('sd = ' + str(sd))

# Compute the autocovaariance of the data
lags = np.arange(-len(EEG) + 1, len(EEG))    # Compute the lags for the full autocovariance vector
                                      # ... and the autocov for L +/- 100 indices
ac = 1 / N * np.correlate(EEG - EEG.mean(), EEG - EEG.mean(), mode='full')
inds = abs(lags) <= 100               # Find the lags that are within 100 time steps

# Plot and save the autocovariance graph
plt.figure(figsize=(12, 6))
plt.plot(lags[inds] * dt, ac[inds])
plt.xlabel('Lag [s]')
plt.ylabel('Autocovariance')
plt.title('Autocovariance of EEG Signal')
plt.savefig('pics/2-3-Autocovariance-1000Hz.png', dpi=500)
plt.grid()
plt.show()

# Compute the power spectrum of the data
xf = fft(EEG - EEG.mean())  # Compute the Fourier transform of the data
Sxx = 2 * dt ** 2 / T * (xf * np.conj(xf))  # Compute the power spectrum
Sxx = Sxx[:N // 2]  # Ignore negative frequencies

df = 1 / T.max()  # Define the frequency resolution
fNQ = 1 / dt / 2  # Define the Nyquist frequency
faxis = np.arange(0, fNQ, df)  # Construct

# Plot and save the power spectrum graph
plt.figure(figsize=(12, 6))
plt.plot(faxis, np.real(Sxx))
plt.xlim([0, 100])
plt.xlabel('Frequency [Hz]')
plt.ylabel(r'Power [$\mu V^2$/Hz]')
plt.title('Power Spectrum of EEG Signal')
plt.savefig('pics/2-3-Power Spectrum-1000Hz.png', dpi=500)
plt.show()

# Plot the power spectrum in log (dB) scale
plt.figure(figsize=(12, 6))
plt.plot(faxis, 20 * np.log10(Sxx))
plt.xlim([0, 70])
plt.ylim([-120, 10])
plt.title('Power Spectrum of EEG Signal (Log Scale)')
plt.savefig('pics/2-3-Power Spectrum-1000Hz(Log Scale).png', dpi=500)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power [dB]')
plt.grid()
plt.show()
