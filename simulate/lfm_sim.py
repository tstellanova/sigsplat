import numpy as np

# Updated parameters
center_freq = 5.405e9  # 5.405 GHz
bandwidth = 50e6  # 50 MHz
prf = 2000  # 2000 Hz pulse repetition frequency

# Calculate the pulse duration from the PRF
pulse_duration = 1 / prf

# Sampling rate for the simulation
sampling_rate = 12e6  # 12 Msps

# Calculate the number of samples
num_samples = int(sampling_rate * pulse_duration)

# Time vector
t = np.arange(num_samples) / sampling_rate

# Generate the chirp signal (LFM)
start_freq = center_freq - bandwidth / 2
end_freq = center_freq + bandwidth / 2
k = (end_freq - start_freq) / pulse_duration  # Chirp rate
phase = 2 * np.pi * (start_freq * t + 0.5 * k * t**2)
chirp_signal = np.exp(1j * phase)

# Normalize the signal to fit within the 8-bit range
max_amplitude = np.max(np.abs(chirp_signal))
chirp_signal_normalized = chirp_signal / max_amplitude

# Convert to signed 8-bit format (cs8)
chirp_signal_cs8 = (chirp_signal_normalized * 127).astype(np.int8)
chirp_signal_cs8_iq = np.zeros(2 * num_samples, dtype=np.int8)
chirp_signal_cs8_iq[0::2] = chirp_signal_cs8.real
chirp_signal_cs8_iq[1::2] = chirp_signal_cs8.imag

# Save to binary file in cs8 format
output_filename = "chirp_signal_sim.cs8"
chirp_signal_cs8_iq.tofile(output_filename)

print(f"Chirp signal saved to {output_filename}")
