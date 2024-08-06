#!/usr/bin/env python
"""
Measure autocorrelation for a signal file
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sigmf
from matplotlib.ticker import StrMethodFormatter
from sigmf import sigmffile, SigMFFile
from scipy.fftpack import fft
from scipy import signal

matplotlib.use('qtagg')


import numpy as np
import matplotlib.pyplot as plt


def read_file_meta(sigfile_obj):
    '''
    Read some commonly-used meta information from sigmf file object
    :param sigfile_obj: SigMF object
    :return:
    '''
    sample_rate_hz = int(sigfile_obj.get_global_field(SigMFFile.SAMPLE_RATE_KEY))
    print(f'sample_rate_hz: {sample_rate_hz}')
    sample_size_bytes = sigfile_obj.get_sample_size()
    print(f'sample size (bytes): {sample_size_bytes}')

    center_freq_hz = sigfile_obj.get_capture_info(0)[SigMFFile.FREQUENCY_KEY]
    half_sampling_rate_hz = sample_rate_hz // 2
    freq_lower_edge_hz = center_freq_hz - half_sampling_rate_hz
    freq_upper_edge_hz = center_freq_hz + half_sampling_rate_hz

    total_samples_guess = sample_rate_hz

    focus_label = None
    first_sample_annotations = sigfile_obj.get_annotations(0)
    for annotation in first_sample_annotations:
        if annotation[SigMFFile.LENGTH_INDEX_KEY] is not None:
            total_samples_guess = int(annotation[SigMFFile.LENGTH_INDEX_KEY])
        if annotation[SigMFFile.FLO_KEY] is not None:
            freq_lower_edge_hz = int(annotation[SigMFFile.FLO_KEY])
        if annotation[SigMFFile.FHI_KEY] is not None:
            freq_upper_edge_hz = int(annotation[SigMFFile.FHI_KEY])
        if annotation[SigMFFile.LABEL_KEY] is not None:
            focus_label = annotation[SigMFFile.LABEL_KEY]

    return center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples_guess, focus_label


def circular_autocorrelation(chunk):
    N = len(chunk)
    chunk_fft = np.fft.fft(chunk, N)
    corr = np.fft.ifft(chunk_fft * np.conj(chunk_fft), N)
    return corr/N

def autocorrelation_fft(signal):
    """
    Compute the autocorrelation of a signal using FFT and IFFT.
    """
    n = len(signal)
    signal_fft = np.fft.fft(signal, n=n*2)
    power_spectrum = signal_fft * np.conj(signal_fft)
    result = np.fft.ifft(power_spectrum)
    result = np.real(result[:n])  # Keep only the non-negative lags and take the real part
    return result / np.arange(n, 0, -1)  # Normalize

def slow_correlation(chunk):
    result = np.correlate(chunk, chunk, mode='full')
    return result[result.size // 2:]

# def slow_correlation(chunk):
#     N = len(chunk)
#     corr = np.correlate(chunk,chunk,mode='full')/N
#     corr = corr[corr.size//2:]
#     return corr

def read_and_plot_n_chunks(
        sigfile_obj,
        title=None,
        ctr_freq_hz=None,
        freqs=None,
        sampling_rate_hz=0,
        sample_start_idx=0,
        chunk_size=64,
        n_chunks=3,
        focus_f_low=None,
        focus_f_high=None,
        focus_label=None):

    n_chunks_read = 0
    end_idx_guess = sample_start_idx + chunk_size*n_chunks
    # freqs = sorted(freqs)

    # if focus_f_low is not None and focus_f_high is not None:
    #     focus_start_idx = np.searchsorted(freqs, focus_f_low)
    #     focus_stop_idx = np.searchsorted(freqs, focus_f_high)
        # print(f"focus_start_idx: {focus_start_idx} focus_stop_idx: {focus_stop_idx}")

    sample_period_ms = (1/sampling_rate_hz)*1E3
    # sample_times_ms = np.arange(0, sample_period_ms * chunk_size, sample_period_ms)
    sample_times_ms = np.linspace(0, sample_period_ms * chunk_size, num=chunk_size)
    correlation_avg_I = np.zeros_like(sample_times_ms, dtype=float)
    correlation_avg_Q = np.zeros_like(sample_times_ms, dtype=float)
    print(f"correlation_avg shape: {correlation_avg_I.shape}")

# typically at the beginning of every cycle theres some spurious high-match correlation--skip it
    initial_skip_inset = chunk_size//100
    print(f"reading {n_chunks} chunks...")
    for sample_idx in range(sample_start_idx, end_idx_guess, chunk_size):
        chunk = sigfile_obj.read_samples(start_index=sample_idx, count=chunk_size)
        # chunk = chunk * np.hamming(len(chunk))
        print(f"correlate {n_chunks_read}")
        corr = circular_autocorrelation(chunk)
        # corr = autocorrelation_fft(chunk)
        # corr = np.correlate(chunk,chunk,mode='full')/chunk.size
        # corr = corr[corr.size//2:]
        # corr = slow_correlation(chunk)/chunk.size
        corr[:initial_skip_inset] = np.median(corr)
        corr[-initial_skip_inset:] = np.median(corr)
        print(f"correlation_avg shape: {correlation_avg_I.shape} corr shape: {corr.shape}")
        # correlation_avg += np.abs(corr)
        correlation_avg_I += np.real(corr)
        correlation_avg_Q += np.abs(np.imag(corr))
        n_chunks_read += 1

    # assert n_chunks_read == n_chunks
    correlation_avg_I /= n_chunks_read
    correlation_avg_Q /= n_chunks_read
    # select the peak with a minimum prominence above immediate surroundings
    print(f"finding peaks...")
    peaks, properties = signal.find_peaks(correlation_avg_I, prominence=0.01)
    peak_sample_time_ms = None
    # print(f"peaks: {peaks} properties: {properties}")
    if properties and properties['prominences'] is not None:
        proms = properties['prominences']
        # print(f"proms: {proms}")
        if proms.size > 0:
            max_prom_idx = proms.argmax()
            max_prom_val = proms[max_prom_idx]
            max_peak_sample_num = peaks[max_prom_idx]
            peak_sample_time_ms = sample_times_ms[max_peak_sample_num]
            peak_cycl_freq = 1E3/(peak_sample_time_ms)
            print(f"max prominence is {max_prom_val:0.6f} at {peak_sample_time_ms} ms ({peak_cycl_freq} Hz) ")

    # plt.figure(figsize=(12, 8))
    subplot_rows = 2
    subplot_row_idx = 0
    fig, axs = plt.subplots(subplot_rows, 1,  figsize=(12, 8))
    if title is None:
        title =  "Sigfile"
    fig.suptitle(title)
    plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
    plt.plot(sample_times_ms, correlation_avg_I)
    if peak_sample_time_ms is not None:
        # plt.plot(sample_times_ms[peaks], correlation_avg[peaks], "x")
        plt.plot(peak_sample_time_ms,correlation_avg_I[max_peak_sample_num],"x")
        # plt.axvline(peak_sample_time_ms, color = "C1")
    plt.grid(True)
    plt.xlabel("Time (ms)")
    plt.ylabel("Avg correlation (I)")

    plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
    plt.plot(sample_times_ms, correlation_avg_Q)
    plt.grid(True)
    plt.ylabel("Avg correlation (Q)")


    # plt.title("Blind Autocorrelation")
    plt.show()



def main():

    parser = argparse.ArgumentParser(description='Analyze SIGMF file for autocorrelation')
    parser.add_argument('src_meta',
                        help="Source sigmf file name eg 'test_sigmf_file' with one of the sigmf file extensions")
    parser.add_argument('--min_lag_ms',dest='min_lag_ms', type=float,
                        help="Minimum repeat time in ms (for autocorrelation calculation)")
    args = parser.parse_args()

    base_data_name = args.src_meta
    print(f'loading file info: {base_data_name}')
    sigfile_obj = sigmffile.fromfile(base_data_name)

    center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples, focus_label \
        = read_file_meta(sigfile_obj)

    print(f" center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples:\n ",
          center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples)

    # This size impacts how likely we are to find an autocorrelation:
    # The longer the size, the more likely that the signal's repeat time will fit within a single chunk
    # For example, GPS L1 C/A repeats every 1 ms, or:
    # 2000 samples at 2 MHz sampling rate,
    # 4000 samples at 4 MHz sampling rate

    chunk_size = 8192

    min_lag_ms = args.min_lag_ms
    if min_lag_ms is not None:
        n_window_samples = (2 * min_lag_ms * 1E-3) * sample_rate_hz
        print(f"n_window_samples: {n_window_samples}")
        chunk_size = int(np.round(n_window_samples))

    freq_resolution = sample_rate_hz / chunk_size

    # fs  >= 2 * fmax
    # calculate optimal chunk size based on Nyquist
    # N = fs / Δf

    print(f"chunk_size: {chunk_size} freq_resolution: {freq_resolution:0.6} Hz")

    n_chunks_guess =  total_samples // chunk_size
    if n_chunks_guess > 20:
        n_chunks_guess = 20
    # plot all the chunks?
    read_and_plot_n_chunks(sigfile_obj,
                           title=base_data_name,
                           freqs=None,
                           sampling_rate_hz=sample_rate_hz,
                           sample_start_idx=0,
                           chunk_size=chunk_size,
                           ctr_freq_hz=center_freq_hz,
                           n_chunks=n_chunks_guess,
                           focus_f_low=freq_lower_edge_hz,
                           focus_f_high=freq_upper_edge_hz,
                           focus_label=focus_label
                           )
    plt.show()




if __name__ == "__main__":
    main()
