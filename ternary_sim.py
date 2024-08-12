#!/usr/bin/env python
"""
Generate a simulated Ternary Coded modulated transmission centered
on a particular (carrier) frequency, similar to that used
for some types of radar
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def generate_ternary_code():
    """
    Generate a Ternary Code Sequence.
    This is based on a sequence published in: https://doi.org/10.1109/TIT.1983.1056707
    Another better source might be: https://doi.org/10.1007/11423461_13
    """
    # A simple ternary sequence example
    perfect_ternary_code = [-1,0,0,+1,0,+1,+1,+1,0,+1,+1,-1,+1,-1,+1,-1,0,+1,+1,+1,+1,-1,-1,+1,+1,-1,-1,-1,
                            +1,-1,-1,-1,0,+1,+1,-1,+1,0,+1,+1,+1,+1,-1,-1,+1,+1,+1,-1,+1,-1,-1,-1,-1,+1,
                            -1,0,+1,+1,-1,+1,-1,-1,-1,+1,0,+1,+1,-1,+1,+1,-1,+1,+1
                            ]
    return perfect_ternary_code

def generate_carrier_wave(center_frequency_hz, sampling_rate_hz, duration_sec):
    """
    Generate the Carrier Wave, which is a simple (co)sine wave centered at desired frequency, with unit amplitude
    :param center_frequency_hz: The desired frequency of the carrier wave
    :param sampling_rate_hz: The desired sampling rate of the returned sequence of samples
    :param duration_sec: The desired duration (length in seconds) of the returned sequence
    :return:
    """
    t = np.arange(0, duration_sec, 1 / sampling_rate_hz)
    carrier_wave = np.cos(2 * np.pi * center_frequency_hz * t)
    return carrier_wave, t

def modulate_with_ternary(carrier_wave, ternary_code):
    """
     Modulate the Carrier Wave with the Ternary Code
    :param carrier_wave:
    :param ternary_code:
    :return: Modulates signal as a sample sequence
    """
    modulated_signal = np.zeros_like(carrier_wave)
    code_length = len(ternary_code)
    samples_per_chip = len(carrier_wave) // code_length

    # TODO this could maybe be made more efficient with numpy, but is fine for an example
    for i, chip in enumerate(ternary_code):
        modulated_signal[i * samples_per_chip: (i + 1) * samples_per_chip] = chip * carrier_wave[i * samples_per_chip: (i + 1) * samples_per_chip]

    return modulated_signal

# Step 4: Measure the Autocorrelation of the Modulated Signal
def autocorrelation(sample_sequence):
    """
    Measure the Autocorrelation of the Modulated Signal
    :param sample_sequence: A sequence of samples
    :return: Normalized autocorrelation
    """
    # result = np.correlate(sample_sequence, sample_sequence, mode='full')
    # result = signal.correlate(sample_sequence, sample_sequence, mode='valid')

    result = signal.correlate(sample_sequence, sample_sequence, mode='full', method='auto')
    result = result[result.size // 2:]/len(sample_sequence)
    print(f"autocorrelation len: {len(result)}")
    return result

# Parameters
n_data_bits = 64
raw_bits = np.random.choice([0, 1], size=n_data_bits)
# ternary_code_length = 17  # Length of the ternary code sequence
carrier_frequency = 10e3   # Carrier frequency in Hz
sampling_rate = 100e3      # Sampling rate in Hz
duration = n_data_bits / (carrier_frequency / 10)  # Duration in seconds

# Generate the ternary code
ternary_code = generate_ternary_code()

# Generate the carrier wave
carrier_wave, time = generate_carrier_wave(carrier_frequency, sampling_rate, duration)

# timestamps used for plotting raw_data
decimated_time = signal.decimate(time, int(len(time) / len(raw_bits)))

# Modulate the carrier wave with the ternary code
modulated_signal = modulate_with_ternary(carrier_wave, ternary_code)

# Measure the autocorrelation of the modulated signal
auto_corr = autocorrelation(modulated_signal)

# Plotting the results
plt.figure(figsize=(12, 8))
subplot_rows = 4
subplot_row_idx = 0

plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
plt.plot(ternary_code)
plt.title('Ternary Code Sequence')
plt.grid(True)

plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
plt.plot(decimated_time, raw_bits)
plt.title('Raw bits')
plt.grid(True)

plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
plt.plot(time, modulated_signal)
plt.title('Modulated Signal')
plt.grid(True)

plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
plt.plot(auto_corr)
plt.title('Autocorrelation of the Modulated Signal')
plt.grid(True)

plt.tight_layout()
plt.show()
