#!/usr/bin/env python
"""
Experiment with searching for Linear Frequency Modulated "chirps" within a signal file
"""
from subprocess import Popen, PIPE, STDOUT
import re
import argparse
import os
import json
from datetime import datetime, timezone

import numpy as np
from sigmf import sigmffile, SigMFFile

from scipy.signal import chirp, correlate
import matplotlib.pyplot as plt

from sigsplat.convert import sigmfun


def search_chirp(iq_signal, fs, f0_range, f1_range, t_range, plot_result=False):
    """
    Searches for a chirp signal in a given I/Q signal using cross-correlation.

    Parameters:
    - iq_signal: The baseband RF I/Q signal (complex-valued numpy array).
    - fs: Sampling frequency of the IQ signal.
    - f0_range: Tuple (min_f0, max_f0) specifying the range of possible start frequencies.
    - f1_range: Tuple (min_f1, max_f1) specifying the range of possible end frequencies.
    - t_range: Tuple (min_t, max_t) specifying the range of possible durations.
    - plot_result: If True, plot the correlation result.

    Returns:
    - best_params: Dictionary with the best found chirp parameters.
    """
    max_corr = 0
    best_params = {}

    for f0 in np.linspace(f0_range[0], f0_range[1], 10):  # Adjust step size for finer search
        for f1 in np.linspace(f1_range[0], f1_range[1], 10):
            if f0 == f1:
                continue # this is just CW, no chirp variation in frequency
            for t in np.linspace(t_range[0], t_range[1], 5):
                t_samples = int(t * fs)
                time = np.linspace(0, t, t_samples)

                # Generate the reference chirp signal
                reference_chirp = chirp(time, f0=f0, f1=f1, t1=t, method='linear')

                # Cross-correlate with the I/Q signal
                correlation = correlate(iq_signal, reference_chirp, mode='valid')
                correlation /= len(reference_chirp)
                correlation = np.abs(correlation) # magnitude of complex correlation
                corr_max = np.max(correlation)

                # Keep track of the best correlation
                if corr_max > max_corr:
                    max_corr = corr_max
                    best_params = {'f0': f0, 'f1': f1, 'duration': t, 'max_corr': max_corr}

                    if plot_result:
                        plt.figure(figsize=(12, 8))
                        plt.plot(correlation)
                        plt.title(f"Cross-correlation (f0={f0}, f1={f1}, duration={t}, max: {max_corr:0.2E})")
                        plt.xlabel("Sample Index")
                        plt.ylabel("Correlation Magnitude")
                        plt.show()

    return best_params



def read_and_search_n_chunks(
        sigfile_obj,
        title=None,
        ctr_freq_hz=None,
        sampling_rate_hz=0,
        sample_start_idx=0,
        chunk_size=64,
        n_chunks=3,
        min_chirp_freq=1E3,
        max_chirp_freq=1E6,
        focus_f_low=None,
        focus_f_high=None,
        focus_label=None,
        chirp_duration_guess=32E-6):

    n_chunks_read = 0
    end_idx_guess = sample_start_idx + chunk_size*n_chunks

    sample_period_ms = (1/sampling_rate_hz)*1E3
    sample_times_ms = np.arange(0, sample_period_ms * chunk_size, sample_period_ms)

    print(f"reading {n_chunks} chunks...")
    for sample_idx in range(sample_start_idx, end_idx_guess, chunk_size):
        print(f"read {sample_idx} .. {sample_idx + chunk_size}  / {end_idx_guess}")
        chunk = sigfile_obj.read_samples(start_index=sample_idx, count=chunk_size)
        n_chunks_read += 1

        # guess at chirp duration
        # chirp_duration_guess = 65E-6  # Duration of the chirp in seconds

        # Define the search ranges
        f0_range = (min_chirp_freq, max_chirp_freq)
        f1_range = (max_chirp_freq, min_chirp_freq)
        t_range = (chirp_duration_guess*0.99, chirp_duration_guess*1.01)   # chip duration range in seconds

        best_chirp = search_chirp(chunk, sampling_rate_hz, f0_range, f1_range, t_range, plot_result=False)
        if bool(best_chirp): # if not empty
            print(f"chunk {n_chunks_read} best_chirp:{best_chirp} rising: { best_chirp['f1'] > best_chirp['f0']}")


def main():
    parser = argparse.ArgumentParser(description='Analyze SIGMF file for linear frequency modulated (LFM) \"chirps\" ')
    parser.add_argument('src_meta',
                        help="Source sigmf file name eg 'test_sigmf_file.sigmf-meta' with the 'sigmf-meta' file extensions")

    parser.add_argument('--pulse_rep_freq_min','-prf_min',dest='prf_min', type=float, default=1E3,
                        help="Minimum pulse repetition frequency (Hz)")
    parser.add_argument('--pulse_rep_freq_max','-prf_max',dest='prf_max', type=float, default=7.6E3,
                        help="Maximum pulse repetition frequency (Hz)")
    # parser.add_argument('--chirp_dur','-cd',dest='chirp_period_sec', type=float,default=73E-6,
    #                     help="Estimate of the duration of each chirp, in seconds")
    parser.add_argument("--min_search_freq","-minf",type=float,default=1.0,required=False,
                        help="Minimum chirp frequency to search, in Hz")
    parser.add_argument("--max_search_freq","-maxf",type=float,default=1.0,required=False,
                        help="Maximum chirp frequency to search, in Hz")

    parser.add_argument("--sample_start_time","-st", dest='sample_start_time',type=float,default=0.0,required=False,
                        help="Start time for valid samples (seconds)")

    args = parser.parse_args()
    prf_min_hz = args.prf_min
    chirp_period_sec = 1/prf_min_hz
    print(f"max chirp period: {chirp_period_sec}")

    # chirp_period_sec = args.chirp_period_sec
    base_data_name = args.src_meta
    print(f'loading file info: {base_data_name}')
    sigfile_obj = sigmffile.fromfile(base_data_name)

    center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples, focus_label \
        = sigmfun.read_file_meta(sigfile_obj)

    print(f" center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples:\n ",
          center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples)

    # 204 us / 73 us =

    # guess at how much time it would take to collect 2 chirps
    resource_period_guess_sec = chirp_period_sec
    resource_period_n_samples = resource_period_guess_sec * sample_rate_hz
    chunk_size = int(resource_period_n_samples // 2)
    print(f"chunk_size calculated: {chunk_size}")

    sample_start_idx = 0
    if args.sample_start_time > 0.0:
        sample_start_idx = args.sample_start_time * sample_rate_hz

    # chunk_size = 16384
    max_n_chunks = 2
    n_chunks_guess =  total_samples // chunk_size
    if n_chunks_guess > max_n_chunks:
        print(f"clamping to {max_n_chunks} chunks")
        n_chunks_guess = max_n_chunks

    chunk_period = chunk_size / sample_rate_hz
    print(f"chunk_period: {chunk_period}")

    # we happen to know this signal starts at 0.00136 seconds
    # sample_start_idx =  int(0.00136  * sample_rate_hz )
    read_and_search_n_chunks(sigfile_obj,
                           title=base_data_name,
                           sampling_rate_hz=sample_rate_hz,
                           sample_start_idx=sample_start_idx,
                           chunk_size=chunk_size,
                           ctr_freq_hz=center_freq_hz,
                           n_chunks=n_chunks_guess,
                           focus_f_low=freq_lower_edge_hz,
                           focus_f_high=freq_upper_edge_hz,
                           focus_label=focus_label
                           )


if __name__ == "__main__":
    main()
