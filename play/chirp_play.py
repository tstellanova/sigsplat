"""
Find the wideband chirps in a noisy signal
"""
import numpy as np
import scipy.signal as signal
import matplotlib.pyplot as plt

# Example parameters
sampling_rate = 1e6  # 1 MHz
center_frequency = 100e3  # 100 kHz
chirp_bandwidth = 50e3  # 50 kHz
chirp_duration = 0.01  # 10 ms
input_signal_mag = 1.0

chirp0_offset = 1000

# Generate a synthetic chirp signal for matching
t = np.arange(0, chirp_duration, 1/sampling_rate)
reference_chirp = input_signal_mag*signal.chirp(t, f0=center_frequency - chirp_bandwidth/2,
                               f1=center_frequency + chirp_bandwidth/2, t1=chirp_duration, method='linear')
reference_chirp = reference_chirp * np.exp(1j * 2 * np.pi * center_frequency * t)  # Upconvert to center frequency
expected_peak0_idx = chirp0_offset + len(reference_chirp)//2
chirp1_offset = chirp0_offset + (len(reference_chirp)*3)//2
expected_peak1_idx = chirp1_offset + len(reference_chirp)//2

# Simulate the input signal (you would replace this with your actual data)
# input_signal = np.zeros(int(0.1 * sampling_rate), dtype=complex)
input_signal = np.random.normal(0, input_signal_mag, int(0.1 * sampling_rate))
input_signal[chirp0_offset:chirp0_offset + len(reference_chirp)] = reference_chirp  # Insert the chirp into the signal
input_signal[chirp1_offset:chirp1_offset+len(reference_chirp)] = reference_chirp  # Insert the chirp into the signal

input_watts = input_signal ** 2
input_avg_watts = np.mean(input_watts)
input_avg_db = 10 * np.log10(input_avg_watts)

mean_noise = 0
target_snr_db = -20
noise_avg_db = input_avg_db - target_snr_db
noise_avg_watts = 10 ** (noise_avg_db / 10)
max_noise_volts = np.sqrt(noise_avg_watts)
noise_volts = np.real(np.random.normal(mean_noise, max_noise_volts, len(input_watts)))
noisy_input_volts = input_signal + noise_volts

# Matched filtering (cross-correlation)
correlation = signal.correlate(noisy_input_volts, reference_chirp, mode='same')/len(noisy_input_volts)

subplot_rows = 3
subplot_row_idx = 0
fig, axs = plt.subplots(subplot_rows, 1,  sharex=True, figsize=(12, 8))
fig.suptitle("Chirp correlation")

plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
plt.plot(input_signal)
plt.grid(True)
plt.ylabel("Signal")

plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
plt.plot(noisy_input_volts)
plt.grid(True)
plt.ylabel("Noisy Signal")

plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
mag_corr = np.abs(correlation)
plt.plot(mag_corr)
plt.grid(True)
plt.ylabel("Correlation mag")


# Detect peaks in the correlation output (indicating chirp presence)
max_corr = np.max(mag_corr)
peaks, _ = signal.find_peaks(np.abs(correlation), prominence=[max_corr * 0.8, max_corr], distance=len(reference_chirp) )
plt.plot(peaks, mag_corr[peaks], "x", color =  "C1")

print(f"Chirp peaks: {peaks} expected: {expected_peak0_idx}, {expected_peak1_idx}")


plt.show()