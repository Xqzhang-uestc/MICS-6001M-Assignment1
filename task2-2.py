from scipy.io import loadmat       # Import function to read data.
from pylab import *                # Import numerical and plotting functions
from scipy.fft import fft, fftfreq # Import FFT functions
# from IPython.lib.display import YouTubeVideo  # Enable YouTube videos
rcParams['figure.figsize']=(12,3)  # Change the default figure size
data = loadmat('./data/02_EEG-1-1second-500Hz.mat')
data.keys()
EEGa = data['EEGa']
EEGb = data['EEGb']
t = data['t'][0]

ntrials = len(EEGa)                             # ... and compute the number of trials.

a_mn = EEGa.mean(0)                               # Compute the mean signal across trials (the ERP).
a_sd = EEGa.std(0)                                # Compute the std of the signal across trials.
a_sdmn = a_sd / sqrt(ntrials)                       # Compute the std of the mean.

plot(t, a_mn, 'k', lw=3)                          # Plot the ERP of condition A,
plot(t, a_mn + 2 * a_sdmn, 'k:', lw=1)              # ... and include the upper CI,
plot(t, a_mn - 2 * a_sdmn, 'k:', lw=1)              # ... and the lower CI.
xlabel('Time [s]')                              # Label the axes,
ylabel('Voltage [$\mu$ V]')
title('ERP of condition A')                     # ... provide a useful title,
show()                                          # ... and show the plot.

b_mn = EEGb.mean(0)                               # Compute the mean signal across trials (the ERP).
b_sd = EEGb.std(0)                                # Compute the std of the signal across trials.
b_sdmn = b_sd / sqrt(ntrials)                       # Compute the std of the mean.

plot(t, b_mn, 'b', lw=3)                          # Plot the ERP of condition B,
plot(t, b_mn + 2 * b_sdmn, 'b:', lw=1)              # ... and include the upper CI,
plot(t, b_mn - 2 * b_sdmn, 'b:', lw=1)              # ... and the lower CI.
xlabel('Time [s]')                              # Label the axes,
ylabel('Voltage [$\mu$ V]')
title('ERP of condition B')                     # ... provide a useful title,
show()                                          # ... and show the plot.

# Compute the FFT of the mean signal for condition A
N = len(a_mn)  # Number of samples
T = t[1] - t[0]  # Sample spacing
yf = fft(a_mn)
xf = fftfreq(N, T)[:N//2]

# Plot the frequency spectrum
plt.figure(figsize=(12, 6))
plt.plot(xf, 2.0/N * np.abs(yf[:N//2]))
plt.title('Frequency Spectrum of Condition A')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')
plt.grid()
plt.show()