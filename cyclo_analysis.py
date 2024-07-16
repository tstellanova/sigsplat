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



def calc_and_plot_caf(sigfile_obj):
    sample_rate_hz = int(sigfile_obj.get_global_field(SigMFFile.SAMPLE_RATE_KEY))
    print(f'sample_rate_hz: {sample_rate_hz}')
    sample_size_bytes = sigfile_obj.get_sample_size()
    print(f'sample size (bytes): {sample_size_bytes}')
    chunk_size = 1024
    total_samples_guess = int(chunk_size*10)

    first_sample_annotations = sigfile_obj.get_annotations(0)
    for annotation in first_sample_annotations:
        if annotation[SigMFFile.LENGTH_INDEX_KEY] is not None:
            total_samples_guess = int(annotation[SigMFFile.LENGTH_INDEX_KEY])

    # 1023 chips
    # code repeats every 1 ms
    # chip rate 1.023 million chips/second
    # 0.000000977517107 seconds/chip
    sps = 1.023E6 # estimated chips / second
    correct_alpha = 1/sps
    taus = np.arange(-100, 101, 1)
    alphas = np.arange(0, 0.5, 0.005)

    plt.figure(figsize=(12, 8))
    plt.xlabel('Alpha')
    plt.ylabel('CAF Power')
    plt.title('CAF')

    n_chunks = 0
    start_idx = 2 * chunk_size
    CAF_avg = np.zeros(len(alphas), dtype=complex)
    for sample_idx in range(start_idx, total_samples_guess, chunk_size):
        chunk = sigfile_obj.read_samples(start_index=sample_idx, count=chunk_size)
        # print(f'chunk shape: {chunk.shape}')

        CAF = np.zeros((len(alphas), len(taus)), dtype=complex)
        for j in range(len(alphas)):
            for i in range(len(taus)):
                CAF[j, i] = np.sum(chunk *
                                   np.conj(np.roll(chunk, taus[i])) *
                                   np.exp(-2j * np.pi * alphas[j] * np.arange(chunk_size)))
        CAF_magnitudes = np.average(np.abs(CAF), axis=1) # at each alpha, calc power in the CAF
        CAF_avg += CAF_magnitudes

        # plt.plot(alphas, CAF_magnitudes,label=f'{n_chunks}')
        n_chunks += 1

    CAF_avg /= n_chunks
    plt.plot(alphas, CAF_avg, label='Avg')
    # plt.xticks(rotation=-45, ha="left")
    # plt.gca().xaxis.set_major_formatter(StrMethodFormatter('{x:,.2f}')) # 2 decimal places
    plt.legend()


def read_file_meta(sigfile_obj):
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

    return center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples_guess


def calc_psd(sigfile_obj, sample_offset=0, sample_rate_hz=2E6, freq_range=None, total_samples=0, chunk_size=10,
             ):
    psd_normalizer = chunk_size*sample_rate_hz
    total_samples_guess = total_samples - sample_offset
    n_chunks_guess = int(total_samples_guess % chunk_size)
    final_idx_guess = n_chunks_guess * chunk_size

    n_chunks = 0
    PSD_avg = np.zeros(chunk_size)
    for sample_idx in range(sample_offset,final_idx_guess,chunk_size):
        chunk = sigfile_obj.read_samples(start_index=sample_idx, count=chunk_size)
        # print(f'chunk shape: {chunk.shape}')
        fft_chunk = np.fft.fft(chunk)
        PSD = np.abs(fft_chunk)**2 / psd_normalizer
        PSD_log = 10.0*np.log10(PSD)
        PSD_shifted = np.fft.fftshift(PSD_log)
        PSD_avg += PSD_shifted
        n_chunks += 1
        # plt.plot(freq_range, PSD_shifted)

    print(f'n_chunks processed: {n_chunks}')
    PSD_avg /= n_chunks
    return PSD_avg




def read_and_plot_n_chunks_psd(sigfile_obj, freqs=None, sampling_rate_hz=0, sample_start_idx=0, chunk_size=64, n_chunks=3):
    n_chunks_read = 0
    end_idx_guess = sample_start_idx + chunk_size*n_chunks

    plt.title('PSD')
    plt.xlabel('Frequency (Hz)')
    power_normalizer = chunk_size * sampling_rate_hz
    PSD_avg = np.zeros(chunk_size, dtype=complex)
    Mag_avg = np.zeros(chunk_size, dtype=complex)
    for sample_idx in range(sample_start_idx, end_idx_guess, chunk_size):
        chunk = sigfile_obj.read_samples(start_index=sample_idx, count=chunk_size)
        # calculate the magnitude metric for this chunk
        samples_mag = np.abs(chunk)
        mag_metric = np.abs(np.fft.fft(samples_mag))
        Mag_avg += mag_metric
        # calculate PSD for this chunk
        fft_data = np.fft.fft(chunk)
        PSD = np.abs(fft_data)**2 / power_normalizer
        PSD_log = 10.0 * np.log10(PSD)
        PSD_shifted = np.fft.fftshift(PSD_log)
        # plt.plot(freqs, PSD_shifted)
        PSD_avg += PSD_shifted
        n_chunks_read += 1
    assert n_chunks_read == n_chunks
    PSD_avg /= n_chunks
    Mag_avg /= n_chunks
    Mag_avg[0] = 0 # null out the DC component

    plt.subplot(3, 1, 1)
    plt.plot(freqs, PSD_avg)
    plt.ylabel('dB')

    plt.subplot(3, 1, 2)
    plt.plot(freqs, Mag_avg)
    plt.ylabel('Magnitude')


def main():

    parser = argparse.ArgumentParser(description='Grab some GNSS data using hackrf_transfer')
    parser.add_argument('src_meta',
                        help="Source sigmf file name eg 'test_sigmf_file' with one of the sigmf file extensions")
    parser.add_argument('--chunksize',dest='chunksize', type=int, default=1024,
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


    fig = plt.figure(figsize=(12, 8))


    # TODO plot all the chunks?
    read_and_plot_n_chunks_psd(sigfile_obj,
                               freqs=chunk_freqs,
                               sampling_rate_hz=sample_rate_hz,
                               sample_start_idx=chunk_size*10,
                               chunk_size=chunk_size,
                               n_chunks=3)


    plt.show()


if __name__ == "__main__":
    main()
