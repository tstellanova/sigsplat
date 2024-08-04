#!/usr/bin/env python
"""
Perform some wavelet analysis on sigmf data files
"""

import argparse
# import time

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pywt
# import sigmf
from sigmf import sigmffile, SigMFFile
# from scipy.signal import welch
# from matplotlib.colors import Normalize, LogNorm, NoNorm
# from scipy import signal, ndimage

matplotlib.use('qtagg')


def replace_zeroes(data):
    min_nonzero = np.min(data[np.nonzero(data)])
    data[data == 0] = min_nonzero
    return data

def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix

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
                      chunk_size=512, n_chunks=3):
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
    sample_spacing_hz = fs / chunk_size
    freqs = np.arange(start=sample_spacing_hz,stop=fs+sample_spacing_hz,step=sample_spacing_hz)
    print(f"freqs: {freqs.shape}  {np.min(freqs)}..{np.max(freqs)}")

    min_scale = 1e-3
    # scales = fs / (freqs * 2 * np.pi)
    scales = pywt.frequency2scale('cmor', freqs/fs)
    assert np.min(scales) >= min_scale

    sample_period_sec = 1/fs
    PSD_avg = np.zeros( (chunk_size, len(freqs)), dtype=float)
    psd_values = []

    end_idx_guess = sample_start_idx + chunk_size*n_chunks
    for sample_idx in range(sample_start_idx, end_idx_guess, chunk_size):
        chunk = sigfile_obj.read_samples(start_index=sample_idx, count=chunk_size)
        if len(chunk) == 0:
            break

        # Perform Continuous Wavelet Transform
        coefs, ret_freqs = pywt.cwt(chunk, scales, 'cmor', sampling_period=1/fs)

        # Calculate PSD (Power Spectral Density)
        # log10 of 0 is NaN
        psd = np.abs(coefs)**2 + np.finfo(float).eps
        # psd = replace_zeroes(psd)
        psd_db = 10 * np.log10(psd)
        # PSD_avg += np.array(psd_db)
        psd_values.append(psd_db)
        if sample_idx == 0:
            print(f"ret_freqs {np.min(ret_freqs)}..{np.max(ret_freqs)}")
            print(f"psd: {psd.shape}")
            plt.figure(figsize=(16, 8))
            times = np.arange(0, chunk_size*sample_period_sec, sample_period_sec)
            print(f"times: {times.shape} {np.min(times)} .. {np.max(times)}")
            print(f"freqs: {freqs.shape} {np.min(freqs)}..{np.max(freqs)}")
            print(f"ret_freqs: {ret_freqs.shape} {np.min(ret_freqs)}..{np.max(ret_freqs)}")

            plt.pcolormesh(times, freqs, psd_db, shading='auto', cmap='jet')
            plt.colorbar(label='PSD')
            plt.xlabel('Time (sec)')
            plt.ylabel('Frequency (Hz)')
            # plt.legend()

            plt.figure(figsize=(16, 8))
            plt.plot(freqs,np.mean(psd_db, axis=0))
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Avg PSD (dB)')
            plt.show()
            break


    # psd_values = np.array(psd_values)
    print(f"psd_values: {len(psd_values)}")
    return psd_values, freqs

def plot_psd(psd_values, frequencies, fs, chunk_size):
    """
    Plot the PSD output from wavelet analysis.

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
    # psd_array = np.array(psd_values) #.T
    #np.concatenate(psd_values, axis=1)  # Concatenate along time axis

    print("plotting...")
    plt.figure(figsize=(16, 8))
    plt.matshow(psd_values)
    plt.ylabel('Chunk')
    plt.xlabel('Frequency (Hz)')
    plt.legend()
    plt.show()

    # plt.pcolormesh(time, frequencies, psd_array, shading='gouraud')
    # plt.matshow(psd_array)
    # plt.imshow(psd_array, aspect='auto')
    # plt.imshow(psd_array, extent=[-1, 1, 1, 31], cmap='PRGn', aspect='auto',
    #            vmax=abs(cwtmatr).max(), vmin=-abs(cwtmatr).max())
    # plt.colorbar(label='Power/Frequency (dB/Hz)')
    # plt.ylabel('Frequency (Hz)')
    # plt.xlabel('Time (s)')
    # plt.title('Time-Frequency Representation of PSD')
    # plt.show()

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

    chunk_size = 1024
    n_chunks_guess =  total_samples // chunk_size
    print(f"n chunks: {n_chunks_guess}")
    psd_values, valid_frequencies  = wavelet_analysis(
        sigfile_obj, fs=sample_rate_hz, fc=center_freq_hz, chunk_size=chunk_size, n_chunks=n_chunks_guess)
    # plot_psd(psd_values, valid_frequencies, sample_rate_hz, chunk_size)



if __name__ == "__main__":
    main()

