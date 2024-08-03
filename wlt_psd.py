#!/usr/bin/env python
"""
Perform some wavelet analysis on sigmf data files
"""

import argparse
import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pywt
import sigmf
from sigmf import sigmffile, SigMFFile
# from scipy.signal import welch
from matplotlib.colors import Normalize, LogNorm, NoNorm
from scipy import signal, ndimage

matplotlib.use('qtagg')


import numpy as np
import matplotlib.pyplot as plt


import numpy as np
import pywt

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

def wavelet_analysis(sigfile_obj, fs, fc, sample_start_idx=0,
                      chunk_size=512, n_chunks=3, frequencies=[100, 200]):
    """
    Perform wavelet analysis on chunks of complex chunk.

    Parameters:
    sigfile_obj: the signal file
    fs (float): Sampling rate.
    fc (float): Center frequency.
    chunk_size (int): Number of samples per chunk.
    frequencies (array-like): Frequencies at which to calculate the PSD.

    Returns:
    psd_values (list of np.ndarray): List of PSD arrays for each chunk.
    """
    min_scale = 1e-3
    scales = fs / (frequencies * 2 * np.pi)
    valid_indices = scales >= min_scale
    valid_scales = scales[valid_indices]
    valid_frequencies = frequencies[valid_indices]
    print(f"valid_frequencies: {valid_frequencies.shape}")
    print(f"valid_scales: {valid_scales.shape}")

    PSD_avg = np.zeros(chunk_size, dtype=float)
    psd_values = []

    end_idx_guess = sample_start_idx + chunk_size*n_chunks
    for sample_idx in range(sample_start_idx, end_idx_guess, chunk_size):
        chunk = sigfile_obj.read_samples(start_index=sample_idx, count=chunk_size)
        if len(chunk) == 0:
            break

        # Perform Continuous Wavelet Transform
        coefs, freqs = pywt.cwt(chunk, valid_scales, 'cmor', sampling_period=1/fs)

        # Calculate PSD (Power Spectral Density)
        sum_samples = np.sum(coefs, axis=1)
        if sample_idx == 0:
            print(f"shape coefs: {coefs.shape} freqs : {freqs.shape} sum_samples: {sum_samples.shape}")

        psd = np.abs(sum_samples)**2
        psd_db = 10 * np.log10(psd)
        psd_values.append(psd_db)

    # psd_values = np.array(psd_values)
    print(f"psd_values: {len(psd_values)}")
    return psd_values, valid_frequencies

def plot_psd(psd_values, frequencies, fs, chunk_size):
    """
    Plot the PSD output from wavelet_analysis.

    Parameters:
    psd_values (list of np.ndarray): List of PSD arrays for each chunk.
    frequencies (array-like): Frequencies at which the PSD was calculated.
    fs (float): Sampling rate.
    chunk_size (int): Number of samples per chunk.
    """
    # Create time axis
    num_chunks = len(psd_values)
    time = np.arange(num_chunks) * (chunk_size / fs)

    # Convert list of PSD arrays to a 2D array
    psd_array = np.array(psd_values).T
    #np.concatenate(psd_values, axis=1)  # Concatenate along time axis

    plt.figure(figsize=(10, 6))
    plt.pcolormesh(time, frequencies, psd_array, shading='gouraud')
    plt.colorbar(label='Power/Frequency (dB/Hz)')
    plt.ylabel('Frequency (Hz)')
    plt.xlabel('Time (s)')
    plt.title('Time-Frequency Representation of PSD')
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Analyze a signal file using wavelets')
    parser.add_argument('src_meta',
                        help="Source sigmf file name eg 'test_sigmf_file' with one of the sigmf file extensions")

    args = parser.parse_args()

    base_data_name = args.src_meta
    print(f'loading file info: {base_data_name}')
    sigfile_obj = sigmffile.fromfile(base_data_name)

    center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples, focus_label \
        = read_file_meta(sigfile_obj)

    print(f"center: {center_freq_hz} samp rate: {sample_rate_hz}")
    sampling_period_sec = 1/sample_rate_hz
    #total_duration = total_samples * sampling_period_sec
    total_duration = 0.1 # second
    total_samples = int(total_duration / sampling_period_sec)
    # sample_times = np.linspace(0, total_duration, total_samples)

    chunk_size = 4096
    half_bandwidth = sample_rate_hz / 2
    # foi = np.linspace(center_freq_hz - half_bandwidth, center_freq_hz + half_bandwidth,num=chunk_size )
    foi = np.linspace(-half_bandwidth, half_bandwidth, num=chunk_size )
    n_chunks_guess =  total_samples // chunk_size
    print(f"n chunks: {n_chunks_guess}")
    print(f"foi: {foi.shape}")
    psd_values, valid_frequencies  = wavelet_analysis(
        sigfile_obj, fs=sample_rate_hz, fc=center_freq_hz, chunk_size=chunk_size, n_chunks=n_chunks_guess, frequencies=foi)
    plot_psd(psd_values, valid_frequencies, sample_rate_hz, chunk_size)



if __name__ == "__main__":
    main()

