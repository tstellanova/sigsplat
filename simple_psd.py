#!/usr/bin/env python
"""
Display average PSD for a whole sigmf file
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sigmf
from matplotlib.ticker import StrMethodFormatter
from sigmf import sigmffile, SigMFFile
from scipy.fftpack import fft

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
    freqs = sorted(freqs)

    if focus_f_low is not None and focus_f_high is not None:
        focus_start_idx = np.searchsorted(freqs, focus_f_low)
        focus_stop_idx = np.searchsorted(freqs, focus_f_high)
        print(f"focus_start_idx: {focus_start_idx} focus_stop_idx: {focus_stop_idx}")

    power_normalizer = chunk_size * sampling_rate_hz
    PSD_avg = np.zeros(chunk_size, dtype=float)

    for sample_idx in range(sample_start_idx, end_idx_guess, chunk_size):
        chunk = sigfile_obj.read_samples(start_index=sample_idx, count=chunk_size)
        # apply a Hamming window function to taper ends of signal to zero
        chunk = chunk * np.hamming(len(chunk))
        # calculate the bare FFT
        chunk_fft = fft(chunk)
        # get magnitude (convert complex to real)
        chunk_fft_mag = np.abs(chunk_fft)

        # chunk_fft = np.abs(chunk_fft[:chunk_size//2])  # Take the positive frequency components only
        chunk_power = chunk_fft_mag**2
        # normalize the power
        chunk_power_norm = chunk_power / power_normalizer
        # convert to dB for PSD view
        chunk_psd_log =  10.0 * np.log10(chunk_power_norm)
        # chunk_psd_shifted = np.fft.fftshift(chunk_psd) # shift zero-freq component to center of spectrum
        PSD_avg += chunk_psd_log
        n_chunks_read += 1

    assert n_chunks_read == n_chunks
    print(f"Averaging over {n_chunks} chunks")
    PSD_avg /= n_chunks
    # clamp endpoints
    median_value = np.median(PSD_avg)
    PSD_avg[0:5] = median_value
    PSD_avg[len(PSD_avg) - 1] = median_value

    max_psd_idx = np.argmax(PSD_avg)
    psd_max_freq = freqs[max_psd_idx]

    if title is None:
        title =  "Sigfile"
    # fig = plt.figure(figsize=(12, 8))
    subplot_rows = 2
    subplot_row_idx = 0
    # fig, axs = plt.subplots(subplot_rows, 1, sharex=True, figsize=(12, 8))
    fig, axs = plt.subplots(subplot_rows, 1,  figsize=(12, 8))
    fig.suptitle(title)
    # fig.subplots_adjust(hspace=0) # remove vspace between plots
    plt.xlabel('Frequency (Hz)')

    subplot_row_idx += 1
    plt.subplot(subplot_rows, 1, subplot_row_idx)
    plt.ylabel('Full PSD (dB)')
    plt.grid(True)
    if ctr_freq_hz is not None:
        plt.axvline(x=ctr_freq_hz, color='lightskyblue')
    if focus_f_low is not None and focus_f_high is not None:
        plt.axvline(x=focus_f_low, color='palegreen')
        plt.axvline(x=focus_f_high, color='palegreen')

    plt.plot(freqs, PSD_avg)
    plt.axvline(x=psd_max_freq, color='violet', label=f"{psd_max_freq}")

    if focus_start_idx is not None:
        subplot_row_idx+=1
        ax = plt.subplot(subplot_rows, 1, subplot_row_idx)
        plt.ylabel('PSD (dB)')
        if focus_label is not None:
            ax.title.set_text(focus_label)
        plt.grid(True)
        plt.axvline(x=ctr_freq_hz, color='lightskyblue',label=f"{ctr_freq_hz/1E6:0.4f}")
        plt.plot(freqs[focus_start_idx:focus_stop_idx], PSD_avg[focus_start_idx:focus_stop_idx])
        plt.axvline(x=psd_max_freq, color='violet', label=f"{psd_max_freq/1E6:0.4f}")

    plt.legend()
    plt.show()

def main():

    parser = argparse.ArgumentParser(description='Analyze SIGMF file')
    parser.add_argument('src_meta',
                        help="Source sigmf file name eg 'test_sigmf_file' with one of the sigmf file extensions")
    parser.add_argument('--fres',dest='freq_resolution', type=float,
                        help="Desired frequency resolution (Hz)")
    parser.add_argument('--min_lag',dest='min_lag', type=int,
                        help="Minimum estimated lag (for autocorrelation calculation)")
    args = parser.parse_args()

    base_data_name = args.src_meta
    print(f'loading file info: {base_data_name}')
    sigfile_obj = sigmffile.fromfile(base_data_name)

    center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples, focus_label\
        = read_file_meta(sigfile_obj)

    print(f" center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples:\n ",
          center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples)


    freq_resolution = args.freq_resolution
    if freq_resolution is None:
        freq_resolution = sample_rate_hz / 2048

    # fs  >= 2 * fmax
    # calculate optimal chunk size based on Nyquist
    # N = fs / Δf
    chunk_size = int(np.ceil(sample_rate_hz / freq_resolution))
    print(f"chunk_size: {chunk_size} freq_resolution: {freq_resolution}")

    sample_spacing_sec = 1 / sample_rate_hz
    chunk_freqs = np.fft.fftfreq(chunk_size, sample_spacing_sec)
    chunk_freqs += center_freq_hz

    n_chunks_guess =  total_samples // chunk_size
    # plot all the chunks?
    read_and_plot_n_chunks(sigfile_obj,
                           title=base_data_name,
                           freqs=chunk_freqs,
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
