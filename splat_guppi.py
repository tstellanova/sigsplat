
import argparse
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from blimpy import GuppiRaw
import os
from scipy.signal import welch

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

def main():
    parser = argparse.ArgumentParser(description='Analyze GUPPI file')
    parser.add_argument('src_raw',
                        help="Source sigmf file name eg 'test_sigmf_file' with one of the sigmf file extensions")
    parser.add_argument('-o', dest='outdir', type=str, default='./img',
                        help='Output directory for PNG files')

    print(f"Hello")
    args = parser.parse_args()
    print("Yippee")
    data_file_name = args.src_raw
    img_out_dir = args.outdir
    print(f"Loading: {data_file_name} , output to: {img_out_dir}")

    r = GuppiRaw(data_file_name)
    n_data_blocks = r.find_n_data_blocks()
    if n_data_blocks > 20:
        n_data_blocks = 20
    heado = r.read_first_header()
    print(f"heado: {heado}")
    obs_center_freq = float(heado['OBSFREQ'])

    n_chans = int(heado['OBSNCHAN'])
    chan_bw = float(heado['CHAN_BW'])
    total_bw = n_chans * chan_bw
    print(f"n_chans: {n_chans} chan_bw: {chan_bw} total_bw: {total_bw} ")
    obs_bw = float(heado['OBSBW'])
    assert obs_bw == total_bw
    half_total_bw = obs_bw / 2
    low_freq_start = obs_center_freq - half_total_bw
    print(f"n_data_blocks: {n_data_blocks}")
    r.reset_index()
    desired_channel_idx = 6
    desired_low_freq = low_freq_start + desired_channel_idx * chan_bw
    desired_high_freq = desired_low_freq + chan_bw
    print(f"desired freq range: {desired_low_freq}... {desired_high_freq}")
    chan_ctr_freq_mhz = (desired_high_freq + desired_low_freq)/2
    chan_sample_rate_mhz = chan_bw
    PSD0_avg = None
    PSD1_avg = None
    n_blocks_read = 0
    for block_idx in range(0,  n_data_blocks):
        block_party = r.read_next_data_block()
        # block_header = block_party[0]
        block_data = block_party[1]
        chan6_data_pol0 = block_data[desired_channel_idx,:,0]
        chan6_data_pol1 = block_data[desired_channel_idx,:,1]
        if PSD0_avg is None:
            PSD0_avg = np.zeros(len(chan6_data_pol0), dtype=float)
            PSD1_avg = np.zeros(len(chan6_data_pol1), dtype=float)

        # chan6_data_pol1 = block_data[desired_channel_idx,:,1]
        # print(f"block_data shape: {block_data.shape} pol0 shape: {chan6_data_pol0.shape} pol1 shape: {chan6_data_pol1.shape}")
        # print(f"pol0 {np.abs(chan6_data_pol0[0])} min {np.min(np.abs(chan6_data_pol0))} .. max {np.max(np.abs(chan6_data_pol0))}")
        # print(f"pol1 {np.abs(chan6_data_pol1[0])} min {np.min(np.abs(chan6_data_pol0))} .. max {np.max(np.abs(chan6_data_pol0))}")

        freqs0, psd0 = calculate_welch_psd(chan6_data_pol0, chan_ctr_freq_mhz, chan_sample_rate_mhz)
        freqs1, psd1 = calculate_welch_psd(chan6_data_pol1, chan_ctr_freq_mhz, chan_sample_rate_mhz)
        # print(f"freqs.shape {freqs0.shape} psd.shape {psd0.shape}")
        psd0_log = 10 * np.log10(np.abs(psd0))
        psd1_log = 10 * np.log10(np.abs(psd1))
        PSD0_avg += psd0_log
        PSD1_avg += psd1_log
        n_blocks_read += 1

    PSD0_avg /= n_blocks_read
    PSD1_avg /= n_blocks_read
    subplot_rows = 2
    subplot_row_idx = 0
    # plt.figure(figsize=(16, 8))
    fig, axs = plt.subplots(subplot_rows, 1,  figsize=(16, 8))
    plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
    plt.axvline(chan_ctr_freq_mhz, color="skyblue")
    plt.plot(freqs0 , PSD0_avg)
    plt.axvline(8420.21645, color="red")
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)
    plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
    plt.axvline(chan_ctr_freq_mhz, color="skyblue")
    plt.plot(freqs1 , PSD1_avg)
    plt.axvline(8420.21645, color="red")
    plt.xlabel('Frequency (MHz)')
    plt.ylabel('Power/Frequency (dB/Hz)')
    plt.grid(True)

    plt.show()


    base_path = os.path.splitext(os.path.basename(data_file_name))[0]
    img_base_path = os.path.join(img_out_dir, base_path)
    # plt.savefig()
    # r.plot_spectrum(filename="%s_spec.png" % img_base_path, plot_db=True, flag_show=True)


if __name__ == "__main__":
    main()
