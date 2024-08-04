"""
Example showing how to calculate the instantaneous phase angle
for a fake DSSS signal similar to L1 C/A GPS
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import hilbert
matplotlib.use('qtagg')

# Parameters
chip_rate = 1.023e6  # chips per second for L1 C/A GPS
sample_rate = 10.23e6  # samples per second (10 times the chip rate)
duration = 0.001  # seconds
num_samples = int(sample_rate * duration)
sample_spacing = 1/sample_rate

# Generate PRN code (simplified for demonstration, typically 1023 chips)
np.random.seed(42)  # For reproducibility
prn_code = np.random.choice([-1, 1], size=int(chip_rate * duration))

# Upsample PRN code to match the sample rate
dsss_signal = np.repeat(prn_code, num_samples // len(prn_code))

# Modulate carrier with PRN code
max_time = float(num_samples * sample_spacing)
t = np.arange(0.0, max_time, sample_spacing)
print(f"timescale: {t}")
# t = np.linspace(0, duration, num_samples, endpoint=False)
# time_factor = np.array((2 * np.pi) * (1.57542e9 * t))
time_factor = np.array(t * 1.57542e9 )
print(f"time_factor: {time_factor}")
carrier = np.cos(time_factor)  # L1 GPS frequency
print(f"carrier: {carrier}")
dsss_modulated_signal = dsss_signal * carrier

# Compute the analytic signal using the Hilbert transform
analytic_signal = hilbert(dsss_modulated_signal)

# Extract the instantaneous phase
instantaneous_phase = np.angle(analytic_signal)

# Plot the results
plt.figure(figsize=(12, 8))
cur_plot_row = 0
num_plot_rows = 4
plt.subplot(num_plot_rows, 1, (cur_plot_row:=cur_plot_row+1))
plt.plot(t, prn_code.repeat(num_samples // len(prn_code)))
plt.title('PRN Code')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(num_plot_rows, 1, (cur_plot_row:=cur_plot_row+1))
plt.plot(t, carrier)
plt.title('Carrier Signal')
plt.yscale('linear')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(num_plot_rows, 1, (cur_plot_row:=cur_plot_row+1))
plt.plot(t, dsss_modulated_signal)
plt.title('DSSS Modulated Signal')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')

plt.subplot(num_plot_rows, 1, (cur_plot_row:=cur_plot_row+1))
plt.plot(t, instantaneous_phase)
plt.title('Instantaneous Phase of DSSS Signal')
plt.xlabel('Time [s]')
plt.ylabel('Phase [rad]')

plt.tight_layout()
plt.show()
