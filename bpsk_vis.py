"""
Visualizing how BPSK is encoded and decoded
"""

import numpy as np
import matplotlib.pyplot as plt
import argparse


# Generate the Fibonacci sequence up to a certain length
def generate_fibonacci_sequence(n, repeats=1):
    fib_sequence = [0, 1]
    while len(fib_sequence) < n:
        fib_sequence.append(fib_sequence[-1] + fib_sequence[-2])
    full_sequence = np.tile(fib_sequence,reps=repeats)
    return full_sequence

# Convert the Fibonacci sequence to binary representation and collect into bytes
def input_sequence_to_bitstream(in_byte_sequence, byte_length=8):
    binary_data = []
    for num in in_byte_sequence:
        binary_str = format(num, '0{}b'.format(byte_length))
        for bit in binary_str:
            binary_data.append(int(bit))
    return np.array(binary_data)


def bpsk_modulate(bit_stream, fc, fs, bit_period_sec=0.01, samples_per_bit=1):
    """
    BPSK modulate the bit stream with carrier frequency fc and sampling frequency fs.
    """
    t = np.arange(0, len(bit_stream) * bit_period_sec, 1 / fs)
    bpsk_signal = np.zeros(len(t))

    for i, bit in enumerate(bit_stream):
        if bit == 0:
            bpsk_signal[i * samples_per_bit:(i + 1) * samples_per_bit] = -np.cos(2 * np.pi * fc * t[i * samples_per_bit:(i + 1) * samples_per_bit])
        else:
            bpsk_signal[i * samples_per_bit:(i + 1) * samples_per_bit] = np.cos(2 * np.pi * fc * t[i * samples_per_bit:(i + 1) * samples_per_bit])

    return bpsk_signal, t

def bpsk_demodulate(bpsk_signal, fc, fs, samples_per_bit=1):
    """
    BPSK demodulate the signal with carrier frequency fc and sampling frequency fs.
    """
    t = np.arange(0, len(bpsk_signal) / fs, 1 / fs)
    carrier = np.cos(2 * np.pi * fc * t)
    demodulated_signal = bpsk_signal * carrier

    # Integrate over each bit duration
    demodulated_bits = []
    for i in range(0, len(demodulated_signal), samples_per_bit):
        bit_chunk = demodulated_signal[i:i + samples_per_bit]
        bit_value = np.trapz(bit_chunk)  # Integrate using the trapezoidal rule
        demodulated_bits.append(1 if bit_value > 0 else 0)

    return demodulated_bits




# Convert binary data back to Fibonacci sequence
def bitstream_to_fibonacci(binary_data, byte_length=8):
    fib_sequence = []
    for i in range(0, len(binary_data), byte_length):
        byte_str = ''.join(str(bit) for bit in binary_data[i:i + byte_length])
        num = int(byte_str, 2)
        fib_sequence.append(num)
    return fib_sequence




def main():

    parser = argparse.ArgumentParser(description='Visualize BPSK encoding and decoding')

    parser.add_argument('--fs',dest='sample_freq_hz', type=float, default=1E3,
                        help="Sampling frequency (Hz)")
    parser.add_argument('--fc',dest='carrier_freq_hz', type=float, default=1E2,
                        help="Carrier / center frequency (Hz)")
    parser.add_argument('--tbit',dest='bit_period', type=float, default=0.01,
                        help="Seconds per bit (bit period)")
    parser.add_argument('--seq_len',dest='seq_len', type=int, default=14,
                        help="Length of input sequence")
    parser.add_argument('--seq_repeats',dest='seq_repeats', type=int, default=3,
                        help="Number of repetitions of input sequence")

    args = parser.parse_args()

    sample_freq_hz = args.sample_freq_hz
    carrier_freq_hz = args.carrier_freq_hz
    bit_period_sec = args.bit_period
    seq_len = args.seq_len
    seq_repeats = args.seq_repeats
    samples_per_bit = int(np.ceil(sample_freq_hz * bit_period_sec))

    input_sequence = generate_fibonacci_sequence(seq_len, repeats=seq_repeats)
    in_bit_stream = input_sequence_to_bitstream(input_sequence)

    # Convert bitstream to BPSK symbols, mixing with carrier
    bpsk_signal, sample_times = bpsk_modulate(
        in_bit_stream,fc=carrier_freq_hz,fs=sample_freq_hz,bit_period_sec=bit_period_sec,samples_per_bit=samples_per_bit)
    # Extract bitstream from mixed carrier-BPSK symbols
    demodulated_bitstream = bpsk_demodulate(
        bpsk_signal,fc=carrier_freq_hz,fs=sample_freq_hz,samples_per_bit=samples_per_bit)

    out_sequence = bitstream_to_fibonacci(demodulated_bitstream)

    # Plot the signals
    plt.figure(figsize=(12, 8))

    plt.subplot(5, 1, 1)
    plt.title('Original Sequence Values')
    plt.stem(input_sequence)

    plt.subplot(5, 1, 2)
    plt.title('In Bitstream')
    plt.plot(in_bit_stream)
    plt.grid(True)
    plt.ylim([-0.2, 1.2])

    plt.subplot(5, 1, 3)
    plt.title('BPSK Signal')
    plt.plot(bpsk_signal)
    plt.grid(True)
    plt.ylim([-1.2, 1.2])

    plt.subplot(5, 1, 4)
    plt.title('Demod Bitstream')
    plt.plot(demodulated_bitstream)
    plt.ylim([-0.2, 1.2])

    plt.subplot(5, 1, 5)
    plt.title('Output Sequence')
    plt.stem(out_sequence)

    plt.tight_layout()

    # plt.figure(figsize=(12, 8))
    # plt.plot(freqs, bpsk_fft)
    # plt.title("Frequency Spectrum of BPSK Modulated Signal")
    # plt.xlabel("Frequency [Hz]")
    # plt.ylabel("Magnitude")
    # plt.grid(True)

    plt.show()




if __name__ == "__main__":
    main()
