import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from scipy.fftpack import fft

# Generate the Fibonacci sequence up to a certain length
def generate_fibonacci_sequence(n, repeats=1):
    fib_sequence = [0, 1]
    while len(fib_sequence) < n:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    full_sequence = np.tile(fib_sequence,reps=repeats)
    return full_sequence


# Convert the Fibonacci sequence to binary representation and collect into bytes
def fibonacci_to_binary_bytes(fib_sequence, byte_length=8):
    binary_data = []
    for num in fib_sequence:
        binary_str = format(num, '0{}b'.format(byte_length))
        for bit in binary_str:
            binary_data.append(int(bit))
    return np.array(binary_data)



# Map binary data to BPSK symbols
def binary_to_bpsk(binary_data):
    bpsk_signal = np.zeros(len(binary_data))
    current_phase = 1  # Start with a phase of 0 degrees (represented by +1)
    for i, bit in enumerate(binary_data):
        if bit == 0:
            current_phase = -current_phase  # 180-degree phase shift
        bpsk_signal[i] = current_phase
    return bpsk_signal

# Design a low-pass filter
def butter_lowpass_filter(data, cutoff, fs, order=5):
    nyquist = 0.5 * fs
    normal_cutoff = cutoff / nyquist
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)
    return y

# Decode the BPSK signal to binary data
def bpsk_to_binary(bpsk_signal):
    binary_data = np.zeros(len(bpsk_signal), dtype=int)
    for i in range(1, len(bpsk_signal)):
        if bpsk_signal[i] != bpsk_signal[i - 1]:
            binary_data[i] = 0  # Phase change detected
        else:
            binary_data[i] = 1  # No phase change
    binary_data[0] = 1 if bpsk_signal[0] == 1 else 0  # Assume the first bit
    return binary_data

# Convert binary data back to Fibonacci sequence
def binary_bytes_to_fibonacci(binary_data, byte_length=8):
    fib_sequence = []
    for i in range(0, len(binary_data), byte_length):
        byte_str = ''.join(str(bit) for bit in binary_data[i:i + byte_length])
        num = int(byte_str, 2)
        fib_sequence.append(num)
    return fib_sequence

# Parameters
fib_length = 14   # Number of Fibonacci numbers to generate
byte_length = 8   # Length of each binary byte
fs = 1000.0       # Sampling frequency
cutoff = 100.0    # Cutoff frequency of the filter

# Generate Fibonacci sequence
fib_sequence = generate_fibonacci_sequence(fib_length,repeats=3)

# Convert Fibonacci sequence to binary data
encoded_binary_data = fibonacci_to_binary_bytes(fib_sequence, byte_length)

# Convert binary data to BPSK symbols
bpsk_signal = binary_to_bpsk(encoded_binary_data)

# Apply low-pass filter to keep the signal centered around zero
# bpsk_modulated = butter_lowpass_filter(bpsk_signal, cutoff, fs)

# Compute FFT
N = len(bpsk_signal)
bpsk_fft = fft(bpsk_signal)
bpsk_fft = np.abs(bpsk_fft[:N//2])  # Take the positive frequency components
bpsk_fft[0] = 0 # DC filter?
freqs = np.fft.fftfreq(N, 1/fs)[:N//2]

# Decode the BPSK signal back to binary data
decoded_binary_data = bpsk_to_binary(bpsk_signal) # bpsk_signal)

# Convert binary data back to Fibonacci sequence
# decoded_fib_sequence = binary_bytes_to_fibonacci(decoded_binary_data, byte_length)
decoded_fib_sequence = binary_bytes_to_fibonacci(decoded_binary_data, byte_length)

# Plot the signals
plt.figure(figsize=(12, 8))

plt.subplot(6, 1, 1)
plt.title('Original Fibonacci Sequence')
plt.stem(fib_sequence)

plt.subplot(6, 1, 2)
plt.title('Encoded Binary Data')
plt.plot(encoded_binary_data)
plt.ylim([-0.2, 1.2])



plt.subplot(6, 1, 3)
plt.title('BPSK Signal')
plt.plot(bpsk_signal)
plt.grid(True)
plt.ylim([-1.2, 1.2])

# plt.subplot(6, 1, 4)
# plt.title('Modulated BPSK Signal')
# plt.plot(bpsk_modulated)
# plt.grid(True)
# plt.ylim([-1.2, 1.2])

plt.subplot(6, 1, 5)
plt.title('Decoded Binary Data')
plt.plot(decoded_binary_data)
plt.ylim([-0.2, 1.2])

plt.subplot(6, 1, 6)
plt.title('Decoded Fibonacci Sequence')
plt.stem(decoded_fib_sequence)

plt.tight_layout()

plt.figure(figsize=(12, 8))
plt.plot(freqs, bpsk_fft)
plt.title("Frequency Spectrum of BPSK Modulated Signal")
plt.xlabel("Frequency [Hz]")
plt.ylabel("Magnitude")
plt.grid(True)

plt.show()


print("Original Fibonacci Sequence:", fib_sequence)
print("Decoded Fibonacci Sequence:", decoded_fib_sequence)
