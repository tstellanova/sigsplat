import argparse
import sigmf
from sigmf import sigmffile, SigMFFile
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from scipy.signal import welch

from sigsplat.convert import sigmfun

matplotlib.use('qtagg')


def calculate_welch_psd(samples, center_frequency, sampling_rate):
    """
    Calculate the power spectral density of complex RF samples.

    Parameters:
    samples (np.array): Array of complex RF samples.
    center_frequency (float): Center frequency of the RF signal in Hz.
    sampling_rate (float): Sampling rate in Hz.
    """
    # Calculate the PSD using Welch's method
    freqs, psd = welch(samples, fs=sampling_rate, nperseg=len(samples), return_onesided=False)

    # Shift the PSD and frequencies to center the plot around the center frequency
    freqs = np.fft.fftshift(freqs)
    psd = np.fft.fftshift(psd)

    # Convert frequencies to be centered around the center frequency
    freqs = freqs + center_frequency

    return freqs, psd



# Example usage
def main():

    parser = argparse.ArgumentParser(description='Analyze SIGMF file')
    parser.add_argument('src_meta',
                        help="Source sigmf file name eg 'test_sigmf_file' with one of the sigmf file extensions")
    # parser.add_argument('--fres',dest='freq_resolution', type=float,
    #                     help="Desired frequency resolution (Hz)")
    # parser.add_argument('--min_lag',dest='min_lag', type=int,
    #                     help="Minimum estimated lag (for autocorrelation calculation)")
    args = parser.parse_args()

    base_data_name = args.src_meta
    print(f'loading file info: {base_data_name}')
    sigfile_obj = sigmffile.fromfile(base_data_name)

    center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples, focus_label \
        = sigmfun.read_file_meta(sigfile_obj)

    print(f" center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples:\n ",
          center_freq_hz, sample_rate_hz, freq_lower_edge_hz, freq_upper_edge_hz, total_samples)
    print(f"total duration (secs): {total_samples / sample_rate_hz}")
    sample_start_idx = 0
    n_chunks_read = 0
    chunk_size = 16384
    n_chunks = total_samples // chunk_size
    end_idx_guess = sample_start_idx + chunk_size*n_chunks
    PSD_avg = np.zeros(chunk_size, dtype=float)

    for sample_idx in range(sample_start_idx, end_idx_guess, chunk_size):
        chunk = sigfile_obj.read_samples(start_index=sample_idx, count=chunk_size)
        n_chunks_read += 1
        freqs, psd = calculate_welch_psd(chunk, center_freq_hz, sample_rate_hz)
        # print(f"freqs.shape {freqs.shape} psd.shape {psd.shape}")
        psd_log = 10 * np.log10(np.abs(psd))
        PSD_avg += psd_log

    # Plot the average PSD
    PSD_avg /= n_chunks_read
    plt.figure(figsize=(16, 8))
    plt.axvline(center_freq_hz, color="skyblue")
    plt.plot(freqs , PSD_avg)
    # plt.axvline(8420.21645E6, color="red")
    plt.title('Power Spectral Density')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()