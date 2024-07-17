#!/usr/bin/env python
"""
Record GNSS data from a particular band/code
using HackRF SDR, and output in sigmf format
"""

import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sigmf
from matplotlib.ticker import StrMethodFormatter
from sigmf import sigmffile, SigMFFile

matplotlib.use('qtagg')


import numpy as np
import matplotlib.pyplot as plt

def autocorrelation_sweep(sigfile_obj, start_index=0, total_samples=int(4E6), chunk_size=8192, max_lag=4096 ):
    lag_limit = max_lag + 1
    autocorr = np.zeros(lag_limit, dtype=np.complex64)

    print(f'autocorrelation_sweep: {start_index} {total_samples} {chunk_size} {max_lag} ')

    for start in range(start_index, total_samples-chunk_size, chunk_size):
        chunk = sigfile_obj.read_samples(start_index=start, count=chunk_size)
        end = min(start + chunk_size, total_samples)
        for lag in range(max_lag + 1):
            if start + lag < total_samples:
                if lag < chunk_size:
                    chunk_lag = chunk[lag:end - start]
                else:
                    chunk_lag = sigfile_obj.read_samples(start + lag, min(chunk_size, total_samples - (start + lag)))
                autocorr[lag] += np.sum(chunk[:len(chunk_lag)] * np.conj(chunk_lag))

    # Normalize the autocorrelation
    autocorr /= total_samples
    return autocorr


def plot_autocorrelation(autocorr):
    plt.figure(figsize=(12, 8))
    lags = np.arange(len(autocorr))

    plt.plot(lags, autocorr.real, label='Real Part')
    plt.plot(lags, autocorr.imag, label='Imaginary Part')

    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    # plt.title('Autocorrelation')
    plt.legend()
    plt.grid(True)

def plot_autocorrelation_mag(autocorr):
    plt.figure(figsize=(12, 8))
    lags = np.arange(len(autocorr))

    plt.plot(lags, np.abs(autocorr), label='Mag')
    # plt.plot(lags, autocorr.imag, label='Imaginary Part')

    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    # plt.title('Autocorrelation')
    plt.legend()
    plt.grid(True)

# # Example usage:
# # Generate a long array of complex64 samples
# long_array = np.random.randn(10000).astype(np.complex64) + 1j * np.random.randn(10000).astype(np.complex64)
#
# # Calculate the autocorrelation for a range of lags (e.g., up to 100 samples)
# autocorr_result = autocorrelation_complex64(long_array, max_lag=100)
#
# # Plot the autocorrelation result
# plot_autocorrelation(autocorr_result)


def calc_and_plot_caf(sigfile_obj, sample_rate_hz, chunk_size=512, total_samples=0):
    '''
    Calculate the Cyclic Autocorrelation Function for the signal samples
    :param sigfile_obj:
    :param sample_rate_hz:
    :param chunk_size:
    :param total_samples:
    :return:
    '''

    # alpha is the cycle frequency: it indicates the periodicity in the statistical properties of the process.
    # In other words, it is the frequency at which the statistical characteristics
    # (such as mean, variance, or autocorrelation) of the process repeat.
    samples_per_symbol = 50 # this is the sample rate for GPS L1 C/A?
    correct_alpha = 1/samples_per_symbol
    max_alpha = 1.25*correct_alpha
    alpha_incr = max_alpha/10
    alphas = np.arange(0, max_alpha, alpha_incr)
    print(f'alphas: {alphas}')

    # The parameter τ (tau) represents the lag or time shift.
    # It is used to measure the time difference between two points in the time series or signal.
    # Here we use a number of samples as a lag
    taus = np.arange(-1024, 1024, 1)

    print(f'taus: {taus}')
    num_calculations = len(taus) * len(alphas)
    print(f"main calculation is {num_calculations}  ")
    # plt.figure(figsize=(12, 8))
    fig, axs = plt.subplots(2, 1, sharex=True, figsize=(12, 8))

    n_chunks = 0
    start_idx = 0
    total_samples = 100*chunk_size
    CAF_basic_avg = np.zeros(len(taus), dtype=complex)
    CAF_mags_avg = np.zeros(len(alphas), dtype=complex)
    print(f'CAF_mags_avg shape: {CAF_mags_avg.shape}')

    for sample_idx in range(start_idx, total_samples, chunk_size):
        chunk = sigfile_obj.read_samples(start_index=sample_idx, count=chunk_size)

        # CAF_2d = np.zeros((len(alphas), len(taus)), dtype=complex)
        # for j in range(len(alphas)):
        #     for i in range(len(taus)):
        #         CAF_2d[j, i] = np.sum(chunk *
        #                            np.conj(np.roll(chunk, taus[i])) *
        #                            np.exp(-2j * np.pi * alphas[j] * np.arange(chunk_size)))
        #
        # CAF_magnitudes = np.average(np.abs(CAF_2d), axis=1) # at each alpha, calc power in the CAF
        # CAF_mags_avg += CAF_magnitudes

        # TODO optimize by re-using one of the above calculations, rather than re-calculating
        # Collect the CAF by tau value, at the "ideal" alpha
        CAF_basic = np.zeros(len(taus), dtype=complex)
        for i in range(len(taus)):
            CAF_basic[i] = np.sum(chunk *
                            np.conj(np.roll(chunk, taus[i])) *
                            np.exp(-2j * np.pi * correct_alpha * np.arange(chunk_size)))
        CAF_basic_avg += np.abs(CAF_basic)

        # plt.plot(alphas, CAF_magnitudes,label=f'{n_chunks}')
        n_chunks += 1

    CAF_basic_avg /= n_chunks
    # CAF_mags_avg /= n_chunks
    max_ele_idx = np.argmax(CAF_mags_avg)
    max_alpha_val = alphas[max_ele_idx]
    print(f"peak alpha val: {max_alpha_val} vs expected: {correct_alpha}")

    subplot_row_idx = 1

    # plt.subplot(2, 1, subplot_row_idx)
    # plt.xlabel('Alpha')
    # plt.ylabel('CAF (real)')
    # plt.axvline(x=correct_alpha, color='palegreen')
    # plt.axvline(x=max_alpha_val, color='salmon')
    # plt.plot(alphas, CAF_mags_avg)
    # # plt.grid()
    # subplot_row_idx += 1

    plt.subplot(2, 1, subplot_row_idx)
    plt.plot(taus, CAF_basic_avg, label='CAF (avg)')
    plt.xlabel('Tau')
    # plt.grid()
    subplot_row_idx += 1


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

    first_sample_annotations = sigfile_obj.get_annotations(0)
    for annotation in first_sample_annotations:
        if annotation[SigMFFile.LENGTH_INDEX_KEY] is not None:
            total_samples_guess = int(annotation[SigMFFile.LENGTH_INDEX_KEY])
        if annotation[SigMFFile.FLO_KEY] is not None:
            freq_lower_edge_hz = int(annotation[SigMFFile.FLO_KEY])
        if annotation[SigMFFile.FHI_KEY] is not None:
            freq_upper_edge_hz = int(annotation[SigMFFile.FHI_KEY])

    print(f'total_sampels_guess: {total_samples_guess}')

    return center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples_guess


def read_and_plot_n_chunks(
        sigfile_obj,
        title=None,
        ctr_freq_hz=None,
        freqs=None,
        sampling_rate_hz=0,
        sample_start_idx=0,
        chunk_size=64,
        n_chunks=3):

    n_chunks_read = 0
    end_idx_guess = sample_start_idx + chunk_size*n_chunks
    freqs = sorted(freqs)

    if title is None:
        title =  "Sigfile"

    # fig = plt.figure(figsize=(12, 8))
    fig, axs = plt.subplots(3, 1, sharex=True, figsize=(12, 8))
    fig.suptitle(title)
    fig.subplots_adjust(hspace=0) # remove vspace between plots
    plt.xlabel('Frequency (Hz)')

    power_normalizer = chunk_size * sampling_rate_hz
    PSD_avg = np.zeros(chunk_size, dtype=complex)
    Magm_avg = np.zeros(chunk_size, dtype=complex)
    for sample_idx in range(sample_start_idx, end_idx_guess, chunk_size):
        chunk = sigfile_obj.read_samples(start_index=sample_idx, count=chunk_size)
        # calculate the magnitude metric for this chunk
        magm = np.abs(np.fft.fft(np.abs(chunk)))
        magm_log = 10.0 * np.log10(magm)
        magm_shifted = np.fft.fftshift(magm_log)
        Magm_avg += magm_shifted

        # calculate the fft of the raw data for this chunk
        fft_data = np.fft.fft(chunk)

        # calculate PSD for this chunk
        PSD = np.abs(fft_data)**2 / power_normalizer
        PSD_log = 10.0 * np.log10(PSD)
        PSD_shifted = np.fft.fftshift(PSD_log)
        # plt.plot(freqs, PSD_shifted)
        PSD_avg += PSD_shifted
        n_chunks_read += 1

    assert n_chunks_read == n_chunks
    ctr_freq_idx = len(freqs) // 2
    PSD_avg /= n_chunks
    PSD_avg[ctr_freq_idx] = np.median(PSD_avg)

    Magm_avg /= n_chunks
    median_magm_avg = np.median(Magm_avg)
    Magm_avg[ctr_freq_idx] = median_magm_avg
    Magm_avg[0] = median_magm_avg

    subplot_row_idx = 1
    plt.subplot(3, 1, subplot_row_idx)
    if ctr_freq_hz is not None:
        plt.axvline(x=ctr_freq_hz, color='palegreen')
    plt.plot(freqs, PSD_avg)
    plt.ylabel('PSD')
    subplot_row_idx+=1

    plt.subplot(3, 1, subplot_row_idx)
    if ctr_freq_hz is not None:
        plt.axvline(x=ctr_freq_hz, color='palegreen')
    plt.plot(freqs, Magm_avg)
    plt.ylabel('Magm')
    subplot_row_idx+=1

    plt.show()


def main():

    parser = argparse.ArgumentParser(description='Grab some GNSS data using hackrf_transfer')
    parser.add_argument('src_meta',
                        help="Source sigmf file name eg 'test_sigmf_file' with one of the sigmf file extensions")
    parser.add_argument('--chunksize',dest='chunksize', type=int, default=1000,
                        help="Number of samples in a chunk")
    args = parser.parse_args()

    base_data_name = args.src_meta
    chunk_size = args.chunksize
    print(f'loading file info: {base_data_name}')
    sigfile_obj = sigmffile.fromfile(base_data_name)

    center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples = read_file_meta(sigfile_obj)

    print(f" center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples:\n ",
          center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples)

    freq_range = np.arange(freq_lower_edge_hz, freq_upper_edge_hz, sample_rate_hz/chunk_size)
    # print(f"freq_range: {freq_range}")
    sample_spacing_sec = 1 / sample_rate_hz
    chunk_freqs = np.fft.fftfreq(chunk_size, sample_spacing_sec)
    chunk_freqs += center_freq_hz
    print(f"freqs {chunk_freqs.shape} f0 {chunk_freqs[0]} f1 {chunk_freqs[1]} .. fN {chunk_freqs[len(chunk_freqs)-1]}")


    n_chunks_guess =  total_samples // chunk_size
    # TODO plot all the chunks?
    # read_and_plot_n_chunks(sigfile_obj,
    #                        title=base_data_name,
    #                        freqs=chunk_freqs,
    #                        sampling_rate_hz=sample_rate_hz,
    #                        sample_start_idx=0,
    #                        chunk_size=chunk_size,
    #                        ctr_freq_hz=center_freq_hz,
    #                        n_chunks=n_chunks_guess)
    # plt.show()

    # calc_and_plot_caf(sigfile_obj,sample_rate_hz=sample_rate_hz,chunk_size=chunk_size,total_samples=total_samples)

    autocorr = autocorrelation_sweep(sigfile_obj,start_index=0,total_samples=total_samples)
    # the first few
    autocorr[0:5] = 0
    plot_autocorrelation_mag(autocorr)
    plt.show()





if __name__ == "__main__":
    main()
