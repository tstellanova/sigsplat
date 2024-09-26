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

from sigsplat.convert import sigmfun

# from scipy.signal import welch
# from matplotlib.colors import Normalize, LogNorm, NoNorm
# from scipy import signal, ndimage

matplotlib.use('qtagg')



def normalize_2d(matrix):
    norm = np.linalg.norm(matrix)
    matrix = matrix/norm  # normalized matrix
    return matrix




def wavelet_analysis(sigfile_obj, fs, fc, sample_start_idx=0,
                      chunk_size=512, n_chunks=3):
    """
    Perform wavelet analysis on chunks of complex chunk.

    Parameters:
    sigfile_obj: the signal file
    fs (float): Sampling rate.
    fc (float): Center frequency.
    chunk_size (int): Number of samples per chunk.

    Returns:
    psd_values (list of np.ndarray): List of PSD arrays for each chunk.
    frequences:  (array-like): Frequencies at which the PSD was calculated
    """
    sample_spacing_hz = fs / chunk_size
    freqs = np.arange(start=sample_spacing_hz,stop=fs+sample_spacing_hz,step=sample_spacing_hz)
    print(f"freqs: {freqs.shape}  {np.min(freqs)}..{np.max(freqs)}")

    min_scale = 1e-3
    # scales = fs / (freqs * 2 * np.pi)
    scales = pywt.frequency2scale('cmor', freqs/fs)
    assert np.min(scales) >= min_scale

    sample_period_sec = 1/fs
    psd_values = []

    n_chunks_read = 0
    end_idx_guess = sample_start_idx + chunk_size*n_chunks
    for sample_idx in range(sample_start_idx, end_idx_guess, chunk_size):
        chunk = sigfile_obj.read_samples(start_index=sample_idx, count=chunk_size)
        if len(chunk) == 0:
            break
        n_chunks_read += 1

        # Perform Continuous Wavelet Transform
        coefs, ret_freqs = pywt.cwt(chunk, scales, 'cmor', sampling_period=1/fs)

        # Calculate PSD (Power Spectral Density)
        # log10 of 0 is NaN
        psd = np.abs(coefs)**2 + np.finfo(float).eps
        psd_db = 10 * np.log10(psd)
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
            plt.legend()

            plt.figure(figsize=(16, 8))
            # The first axis of coefs corresponds to the scales
            lucky_charms = np.median(psd_db, axis=0)
            print(f"lucky {np.min(lucky_charms)}..{np.max(lucky_charms)}")
            plt.plot(freqs, lucky_charms)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Avg PSD (dB)')
            # plt.axvline(x=fc, color='lightskyblue',label=f"{fc:0.4f}")
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
        = sigmfun.read_file_meta(sigfile_obj)
    print(f"center: {center_freq_hz} samp rate: {sample_rate_hz}")

    chunk_size = 1024
    n_chunks_guess =  total_samples // chunk_size
    print(f"n chunks: {n_chunks_guess}")
    psd_values, valid_frequencies  = wavelet_analysis(
        sigfile_obj, fs=sample_rate_hz, fc=center_freq_hz, chunk_size=chunk_size, n_chunks=n_chunks_guess)
    # plot_psd(psd_values, valid_frequencies, sample_rate_hz, chunk_size)



if __name__ == "__main__":
    main()

