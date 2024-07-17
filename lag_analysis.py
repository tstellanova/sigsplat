gi#!/usr/bin/env python
"""
Read samples from sigmf file and analyze for autocorrelation
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

def autocorrelation_sweep(
        sigfile_obj, start_index=0, total_samples=int(4E6), chunk_size=8192, min_lag=0, max_lag=4096 ):
    lag_limit = max_lag + 1
    autocorr = np.zeros(lag_limit, dtype=np.complex64)
    lags = range(min_lag, max_lag + 1)
    print(f'autocorrelation_sweep: start {start_index} total {total_samples} chunk_size {chunk_size} min_lag {min_lag } max_lag {max_lag} ')

    for start in range(start_index, total_samples-chunk_size, chunk_size):
        # print(f'autocorr {start} / {total_samples}')
        chunk = sigfile_obj.read_samples(start_index=start, count=chunk_size)
        end = min(start + chunk_size, total_samples)
        for lag in lags:
            if start + lag < total_samples:
                if lag < chunk_size:
                    chunk_lag = chunk[lag:end - start]
                else:
                    chunk_lag = sigfile_obj.read_samples(start + lag, min(chunk_size, total_samples - (start + lag)))
                autocorr[lag] += np.sum(chunk[:len(chunk_lag)] * np.conj(chunk_lag))

    # Normalize the autocorrelation
    autocorr /= total_samples
    return lags,autocorr



def plot_autocorrelation(autocorr, lags=None, title=None):
    fig = plt.figure(figsize=(12, 8))
    if title is not None:
        fig.suptitle(title)
    if lags is None:
        lags = np.arange(len(autocorr))
    n_lags = len(lags)
    subset = autocorr[-n_lags:]
    print(f'{subset.shape}')

    plt.plot(lags, np.abs(subset), label='Mag', color='salmon')
    plt.plot(lags, subset.real, label='Re', color='palegreen')
    plt.plot(lags, subset.imag, label='Im', color='paleturquoise')

    plt.xlabel('Lag')
    plt.ylabel('Autocorrelation')
    # plt.title('Autocorrelation')
    plt.legend()
    plt.grid(True)



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

    return center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples_guess




def main():

    parser = argparse.ArgumentParser(description='Grab some GNSS data using hackrf_transfer')
    parser.add_argument('src_meta',
                        help="Source sigmf file name eg 'test_sigmf_file' with one of the sigmf file extensions")
    parser.add_argument('--chunk_size',dest='chunk_size', type=int, default=2048,
                        help="Number of samples in a chunk")
    parser.add_argument('--max_lag',dest='max_lag', type=int,
                        help="Maximum estimated lag (for autocorrelation calculation)")
    parser.add_argument('--min_lag',dest='min_lag', type=int,
                        help="Minimum estimated lag (for autocorrelation calculation)")
    args = parser.parse_args()

    base_data_name = args.src_meta
    chunk_size = args.chunk_size
    max_lag = args.max_lag
    min_lag = args.min_lag
    print(f'loading file info: {base_data_name}')
    sigfile_obj = sigmffile.fromfile(base_data_name)

    center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples = read_file_meta(sigfile_obj)

    print(f" center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples:\n ",
          center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples)


    sample_rate_mhz_int = int(sample_rate_hz / 1E6)
    max_lag_guess = sample_rate_mhz_int * 2000
    min_lag_guess = 0
    if max_lag is not None:
        max_lag_guess = max_lag
    if min_lag is not None:
        min_lag_guess = min_lag
    lags,autocorr = autocorrelation_sweep(
        sigfile_obj,start_index=0,total_samples=total_samples, chunk_size=chunk_size, min_lag=min_lag_guess, max_lag=max_lag_guess)
    if max_lag is None:
        # the first few autocorrelation results are rarely interesting
        autocorr[0:25] = 0
    plot_autocorrelation(autocorr, lags=lags , title=base_data_name)
    plt.show()





if __name__ == "__main__":
    main()
