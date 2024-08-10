#!/usr/bin/env python
"""
Generate a simulated Barker Code transmission centered
on a particular (carrier) frequency, similar to that used
for planetary radar.
"""
import numpy as np
import re
import argparse
import os
import json
from datetime import datetime, timezone
import matplotlib.pyplot as plt

import sigmf
from sigmf import SigMFFile

GOLDSTONE_BIT_PERIOD_SEC = 1E-6 # Duration of a single bit in Goldstone transmission

def generate_barker_code_13_bit():
    # Generate 13-bit Barker code sequence
    barker_code = np.array([1, 1, 1, 1, 1, -1, -1, 1, 1, -1, 1, -1, 1])
    return barker_code


def simulate_goldstone_radar(fs, fc, duration):
    """

    :param fs: Sampling frequency (Hz)
    :param fc: Center (carrier) frequency (Hz)
    :param duration to generate, in seconds
    :return: ndarray with simulated Barker-coded signal
    """
    # Generate the Barker code
    barker_code = generate_barker_code_13_bit()

    # Length of the Barker code
    code_length = len(barker_code)
    assert 13 == code_length

    T_bit = GOLDSTONE_BIT_PERIOD_SEC   # Duration of each Barker code bit in seconds (1 µs for Goldstone), equiv to 1 Mbps
    # total_msg_num_bits = duration / T_bit
    # Calculate the number of samples per bit based on T_bit
    samples_per_bit = int(T_bit * fs)
    print(f"samples_per_bit: {samples_per_bit}")

    # Total duration of the signal
    total_samples = duration * fs
    total_bits_max = total_samples / samples_per_bit
    code_repeats = int(np.ceil(total_bits_max / code_length))
    total_samples = code_repeats * code_length * samples_per_bit
    print(f"code_repeats: {code_repeats} total_samples: {total_samples}")

    # Time vector for the entire transmission duration
    t = np.arange(0, total_samples) / fs
    barker_sequence = np.repeat(barker_code, samples_per_bit)
    print(f"barker_sequence shape: {barker_sequence.shape}  {barker_sequence}")

    # Generate the carrier signal
    carrier_signal = np.cos(2 * np.pi * fc * t)
    print(f"carrier_signal: {carrier_signal.shape}")
    # Generate the modulation signal based on the Barker code
    modulation_signal = np.tile(barker_sequence, code_repeats)
    print(f"modulation_sig shape: {modulation_signal.shape} ")

    # Modulate the carrier signal with the Barker code
    radar_signal = carrier_signal * modulation_signal

    return radar_signal, modulation_signal, carrier_signal, t


def main():
    parser = argparse.ArgumentParser(description='Grab sosme GNSS data using a remote Pluto SDR over IP')
    parser.add_argument('--freq_center','-fc', dest='freq_center_hz',type=float, default=1420.4000E6,
                        help='Center (carrier) frequency to use (Hz)')
    parser.add_argument('--sampling_rate','-fs', dest='sampling_rate_hz',type=float, default=2E6,
                        help='Samping rate to use (Hz)')
    parser.add_argument("--out_path",dest='out_path',default='./',
                        help="Directory path to place output files" )
    parser.add_argument('--duration', '-d',  dest="duration_secs", type=float,
                        default=10*13*GOLDSTONE_BIT_PERIOD_SEC,
                        help='Duration to generate, in seconds')

    args = parser.parse_args()
    out_path = args.out_path
    freq_center_hz = args.freq_center_hz
    sampling_rate_hz = args.sampling_rate_hz
    duration_secs = args.duration_secs

    if not os.path.isdir(out_path):
        print(f"out_path {out_path} does not exist")
        return -1

    radar_signal, modulation_signal, carrier_signal, times \
        = simulate_goldstone_radar(sampling_rate_hz, freq_center_hz, duration_secs)
    # print(f"times: {times} signal: {radar_signal}")

    subplot_rows = 3
    subplot_row_idx = 0
    fig, axs = plt.subplots(subplot_rows, 1,  sharex=True, figsize=(12, 8))
    fig.suptitle("Goldstone Barker Simulation")
    fig.subplots_adjust(hspace=0) # remove vspace between plots
    plt.xlabel("Time (sec)")

    plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
    plt.plot(times, carrier_signal, color='g')
    plt.grid(True)
    plt.ylabel("Carrier")

    plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
    plt.plot(times, modulation_signal, color='g')
    plt.grid(True)
    plt.ylabel("Modulation")

    plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
    plt.plot(times, radar_signal, color='g')
    plt.grid(True)
    plt.ylabel("Radar")

    plt.show()


if __name__ == "__main__":
    main()