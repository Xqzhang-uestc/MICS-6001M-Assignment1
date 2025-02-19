import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch, spectrogram
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Parameters
f = 10  # base frequency in Hz
sampling_rate = 200  # sampling rate in Hz
duration = 1  # duration of each sine wave in seconds
total_duration = 5  # total duration of the signal in seconds

# Time vector for one second
t = np.linspace(0, duration, int(sampling_rate * duration), endpoint=False)

# Generate the sine waves
sine_waves = []
for n in range(1, 6):
    sine_wave = np.sin(2 * np.pi * n * f * t)
    sine_waves.append(sine_wave)

# Concatenate the sine waves to form the final signal
final_signal = np.concatenate(sine_waves)

# Time vector for the final signal
t_final = np.linspace(0, total_duration, int(sampling_rate * total_duration), endpoint=False)

# Plot the final signal
plt.figure(figsize=(10, 4))
plt.plot(t_final, final_signal)
plt.title('Sequential Sine Wave Signals')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid(True)
plt.show()

# Power spectrum (power density graph)
frequencies, power_density = welch(final_signal, fs=sampling_rate, nperseg=1024)
plt.figure(figsize=(10, 4))
plt.semilogy(frequencies, power_density)
plt.title('Power Spectrum Density')
plt.xlabel('Frequency [Hz]')
plt.ylabel('Power Density')
plt.grid(True)
plt.show()

# Spectrogram
frequencies, times, Sxx = spectrogram(final_signal, fs=sampling_rate)
plt.figure(figsize=(10, 4))
plt.pcolormesh(times, frequencies, 10 * np.log10(Sxx), shading='gouraud')
plt.title('Spectrogram')
plt.ylabel('Frequency [Hz]')
plt.xlabel('Time [s]')
plt.colorbar(label='Power/Frequency (dB/Hz)')
plt.show()

# ACF and PACF for the first one-second (frequency 10Hz)
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plot_acf(sine_waves[0], lags=50, ax=plt.gca())
plt.title('ACF of 10Hz Sine Wave')
plt.subplot(2, 1, 2)
plot_pacf(sine_waves[0], lags=50, ax=plt.gca())
plt.title('PACF of 10Hz Sine Wave')
plt.tight_layout()
plt.show()

# ACF and PACF for the second one-second (frequency 20Hz)
plt.figure(figsize=(10, 8))
plt.subplot(2, 1, 1)
plot_acf(sine_waves[1], lags=50, ax=plt.gca())
plt.title('ACF of 20Hz Sine Wave')
plt.subplot(2, 1, 2)
plot_pacf(sine_waves[1], lags=50, ax=plt.gca())
plt.title('PACF of 20Hz Sine Wave')
plt.tight_layout()
plt.show()