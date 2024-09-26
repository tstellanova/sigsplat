"""
Process a raw GUPPI file
Look for changes in power over time
"""
import argparse

import blimpy
import matplotlib
import matplotlib.pyplot as plt
import sys

import numpy as np
from blimpy import GuppiRaw
import os

from scipy import ndimage

from time import perf_counter

from sigsplat.convert import fbank, spectralize
matplotlib.use('qtagg')






def process_dual_pol_block(chan_pol_a=None, chan_pol_b=None, n_freq_bins=1024, n_vps_rows_per_block=1):
    if chan_pol_a is None:
        chan_pol_a = []
    if chan_pol_b is None:
        chan_pol_b = []

    all_block_vps = []
    # all_block_corrs = []
    # For 8 bit samples, each complex sample consists of two bytes:
    # one byte for real followed by one byte for imaginary.
    chan_pol_a_iq = chan_pol_a.flatten()
    chan_pol_b_iq = chan_pol_b.flatten()

    # TODO can we jump straight to view complex64 without intermediate astype?
    chan_pol_a_view = chan_pol_a_iq.astype(np.float32).view('complex64')
    # print(f"chan_pol_a_view {chan_pol_a_view.shape}")
    chan_pol_b_view = chan_pol_b_iq.astype(np.float32).view('complex64')
    # assert samples_per_chan_per_block == len(chan_pol_a_view)

    # treat the arbitrary block size as a series of sample rows
    newshape = (n_vps_rows_per_block, n_freq_bins)
    # print(f"reshape pol_views: {chan_pol_a_view.shape} to {newshape}")
    chan_pol_a_view = np.reshape(chan_pol_a_view, newshape)
    chan_pol_b_view = np.reshape(chan_pol_b_view, newshape)
    # print(f"chan_pol_a_view {chan_pol_a_view.shape}")

    # window_filter = np.hamming(n_freq_bins)
    window_filter = np.hanning(n_freq_bins)

    # # Define the taper length (number of points to taper at the beginning and end)
    # taper_length = int(n_freq_bins // 10)  # 10% taper at both ends
    #
    # # Create the taper function (cosine taper)
    # taper = np.ones(n_freq_bins)
    # taper[:taper_length] = 0.5 * (1 - np.cos(np.linspace(0, np.pi, taper_length)))
    # taper[-taper_length:] = 0.5 * (1 - np.cos(np.linspace(np.pi, 0, taper_length)))
    # window_filter = taper

    print(f"Calc VPS for {n_vps_rows_per_block} block rows...")
    perf_start = perf_counter()
    for psd_row_idx in range(0, n_vps_rows_per_block):
        # apply window filter to fix circularity
        chunk_a = chan_pol_a_view[psd_row_idx]
        chunk_a = chunk_a * window_filter
        chunk_b = chan_pol_b_view[psd_row_idx]
        chunk_b = chunk_b * window_filter
        row_pol_a_fft = np.fft.fft(chunk_a, n=n_freq_bins)
        row_pol_b_fft = np.fft.fft(chunk_b, n=n_freq_bins)
        fft_res_len = len(row_pol_a_fft)
        if fft_res_len != n_freq_bins:
            print(f"original N: {n_freq_bins} / fft_res_len: {fft_res_len}")
            row_pol_a_fft = row_pol_a_fft[:n_freq_bins]
            row_pol_b_fft = row_pol_b_fft[:n_freq_bins]

        # Calculate Voltage Power Spectrum from (complex) FFT results
        chan_pol_a_vps = (row_pol_a_fft * np.conj(row_pol_a_fft)) / n_freq_bins
        chan_pol_b_vps = (row_pol_b_fft * np.conj(row_pol_b_fft)) / n_freq_bins

        # calculate the elementwise correlation between polarizations (noise should be uncorrelated)
        chan_pol_a_vps_norm = normalize_data(chan_pol_a_vps)
        chan_pol_b_vps_norm = normalize_data(chan_pol_b_vps)
        raw_corr = np.multiply(chan_pol_a_vps_norm, chan_pol_b_vps_norm)
        # use "elementwise correlation" as a threshold
        cur_vps_corr = np.where(raw_corr < 0.7, 0.0, 1.0)
        # avg_corr = np.average(np.abs(cur_vps_corr))
        # all_block_corrs.append(avg_corr)

        # we sum the power from both polarizations and scale by how correlated they are
        sum_vps = chan_pol_a_vps + chan_pol_b_vps
        cur_vps = np.abs(np.multiply(sum_vps,cur_vps_corr))
        all_block_vps.append(cur_vps)

    # all_corrs = np.asarray(all_block_corrs)
    # print(f"block corr min: {np.min(all_corrs):0.4f} max: {np.max(all_corrs):0.4f} avg: {np.average(all_corrs):0.4f}")
    res_all_block_vps = np.asarray(all_block_vps)
    print(f"elapsed: {perf_counter()  - perf_start:0.3f} seconds")

    return res_all_block_vps

def process_chan_over_all_blocks(raw_reader=None, chan_idx=0, n_blocks=1, n_freq_bins=1024):
    focus_chan_vps_diffs = []
    all_focus_chan_vps = []
    prior_vps = None
    n_psd_rows_per_block = None
    vps_row_idx = 0
    print(f"Processing chan {chan_idx} over n_data_blocks: {n_blocks}")
    for block_idx in range(0, n_blocks):
        print(f"read block {block_idx} / {n_blocks} ...")
        perf_start = perf_counter()
        _block_hdr, data_pol_a, data_pol_b = raw_reader.read_next_data_block_int8()
        print(f"elapsed: {perf_counter()  - perf_start:0.3f} seconds")

        chan_pol_a = data_pol_a[chan_idx]
        chan_pol_b = data_pol_b[chan_idx]

        if n_psd_rows_per_block is None:
            print(f"chan_pol_a : {chan_pol_a.shape}")
            samples_per_chan_per_block = chan_pol_a.shape[0]
            n_psd_rows_per_block = int(samples_per_chan_per_block // n_freq_bins)
            n_psd_rows = int(n_blocks * n_psd_rows_per_block)
            print(f"n_blocks: {n_blocks} samples_per_chan_per_block: {samples_per_chan_per_block} "
                  f"n_freq_bins: {n_freq_bins} n_psd_rows_per_block: {n_psd_rows_per_block} n_psd_rows: {n_psd_rows}")

        all_block_vps = process_dual_pol_block(chan_pol_a, chan_pol_b, n_freq_bins, n_psd_rows_per_block)
        # print(f"all_block_vps {all_block_vps.shape}")

        print(f"Calculate {len(all_block_vps)} VPS diffs...")
        perf_start = perf_counter()
        for cur_vps in all_block_vps:
            # print(f"cur_vps {cur_vps.shape}")
            all_focus_chan_vps.append(cur_vps)
            if prior_vps is None:
                # the zeroth psd_diff will always be zero
                prior_vps = cur_vps
                vps_row_idx += 1
                continue
            # print(f"cur_vps {cur_vps.shape} prior_vps {prior_vps.shape}")
            cur_vps_diff = np.subtract(cur_vps, prior_vps)
            # print(f"cur_vps_diff {cur_vps_diff.shape} focus_chan_vps_diffs[vps_row_idx] {focus_chan_vps_diffs[vps_row_idx].shape}")
            focus_chan_vps_diffs.append(cur_vps_diff)
            prior_vps = cur_vps
            vps_row_idx += 1
        print(f"elapsed: {perf_counter()  - perf_start:0.3f} seconds")

    all_chan_vps = np.asarray(all_focus_chan_vps)
    all_chan_vps_diffs = np.asarray(focus_chan_vps_diffs)
    # print(f"all_chan_vps_diffs {all_chan_vps_diffs}")

    return all_chan_vps, all_chan_vps_diffs

def main():
    parser = argparse.ArgumentParser(description='Analyze GUPPI file')
    parser.add_argument('src_path', nargs='?',
                        help="Source raw file path eg 'blc21_guppi.raw' with the '.raw' file extension",
                        default="../../baseband/bloda/"
                                "guppi_58198_27514_685364_J0835-4510_B3_0001.0000.raw"
                                # "blc4_2bit_guppi_57432_24865_PSR_J1136+1551_0002.0001.raw"
                        )
    parser.add_argument('-o', dest='outdir', type=str, default='./data',
                        help='Output directory processed files')

    args = parser.parse_args()
    data_file_name = args.src_path
    out_dir = args.outdir
    print(f"Loading: {data_file_name} , output to: {out_dir}")

    r = GuppiRaw(data_file_name)
    # Seek through the file to find how many data blocks there are in the file
    max_blocks = 4
    n_data_blocks = r.find_n_data_blocks()
    print(f"total n_data_blocks: {n_data_blocks}")
    if n_data_blocks > max_blocks:
        print(f"clamping to {max_blocks} blocks")
        n_data_blocks = max_blocks

    # reads the first header and then seek to file offset zero
    first_header = r.read_first_header()
    print(f"first_header: {first_header}")
    highest_obs_freq_mhz = float(first_header['OBSFREQ'])
    print(f"highest_obs_freq_mhz {highest_obs_freq_mhz} MHz")
    n_coarse_chans = int(first_header['OBSNCHAN'])
    # chan_bw_mhz = float(first_header['CHAN_BW'])

    # Per davidm, 'CHAN_BW': 2.9296875 (MHz) and 'TBIN': 3.41333333333e-07 should be inverses of each other
    time_bin = float(first_header['TBIN'])
    print(f"TBIN: {time_bin:0.3e}")
    # verify_time_bin = 1 / (np.abs(chan_bw_mhz) * 1E6)
    # print(f"verify_tbin - time_bin: {verify_time_bin - time_bin:0.3e}")
    # assert verify_time_bin == time_bin

    total_obs_bw_mhz = np.abs(float(first_header['OBSBW']))

    # total_obs_bw_mhz = n_coarse_chans * np.abs(chan_bw_mhz)
    # obs_bw_mhz = np.abs(float(first_header['OBSBW']))
    # print(f"n_coarse_chans: {n_coarse_chans} chan_bw_mhz: {chan_bw_mhz} total_obs_bw_mhz: {total_obs_bw_mhz}")
    # # print(f"obs_bw_mhz: {obs_bw_mhz} total_obs_bw_mhz: {total_obs_bw_mhz} ")
    # assert obs_bw_mhz == total_obs_bw_mhz

    n_pols = int(first_header['NPOL'])
    n_bits = int(first_header['NBITS'])
    block_size = int(first_header['BLOCSIZE'])
    # NTIME = BLOCSIZE * 8 / (2 * NPOL * NCHAN * NBITS)
    ntime_calc = block_size * 8 / (2 * n_pols * n_coarse_chans * n_bits)
    print(f"NTIME: {ntime_calc} BLOCSIZE: {block_size} NPOL: {n_pols} NCHAN: {n_coarse_chans} NBITS: {n_bits}")

    if n_pols > 2:
        n_pols = np.sqrt(n_pols)

    lowest_obs_freq_mhz = highest_obs_freq_mhz - total_obs_bw_mhz
    print(f"avail freq range: {lowest_obs_freq_mhz}... {highest_obs_freq_mhz}")

    # first_header: {
    # 'BACKEND': 'GUPPI', 'TELESCOP': 'GBT', 'OBSERVER': 'Andrew Siemion', 'PROJID': 'AGBT16A_999_17',
    # 'FRONTEND': 'Rcvr8_10', 'NRCVR': 2, 'FD_POLN': 'CIRC',
    # 'SRC_NAME': 'VOYAGER1',
    # 'OBSFREQ': 9093.75,  'OBSBW': 187.5, 'CHAN_BW': 2.9296875, 'TBIN': 3.41333333333e-07,
    # 'BASE_BW': 1450.0,
    # 'NBITS': 2,
    # 'TRK_MODE': 'TRACK', 'RA_STR': '17:12:16.1760', 'RA': 258.0674, 'DEC_STR': '+11:57:46.4400', 'DEC': 11.9629,
    # 'LST': 81951, 'AZ': -84.6901, 'ZA': 77.3374, 'BMAJ': 0.02290245676433732, 'BMIN': 0.02290245676433732,
    # 'DAQPULSE': 'Sat Jan  9 15:49:54 2016', 'DAQSTATE': 'running', 'NBITS': 2, 'OFFSET0': 0.0, 'OFFSET1': 0.0,
    # 'OFFSET2': 0.0, 'OFFSET3': 0.0, 'BANKNAM': 'BANKD', 'TFOLD': 0, 'DS_FREQ': 1, 'DS_TIME': 1, 'FFTLEN': 32768,
    # 'NBIN': 256, 'OBSNCHAN': 64, 'SCALE0': 1.0, 'SCALE1': 1.0, 'DATAHOST': '10.17.0.142',
    # 'SCALE3': 1.0, 'NPOL': 4, 'POL_TYPE': 'AABBCRCI', 'BANKNUM': 3, 'DATAPORT': 60000, 'ONLY_I': 0, 'CAL_DCYC': 0.5,
    # 'DIRECTIO': 0, 'BLOCSIZE': 33062912, 'ACC_LEN': 1, 'CAL_MODE': 'OFF', 'OVERLAP': 0, 'OBS_MODE': 'RAW',
    # 'CAL_FREQ': 'unspecified', 'DATADIR': '/datax/dibas', 'PFB_OVER': 12, 'SCANLEN': 300.0,
    # 'PARFILE': '/opt/dibas/etc/config/example.par',
    # 'SCALE2': 1.0, 'BINDHOST': 'eth4',
    # 'PKTFMT': '1SFA',  'CHAN_DM': 0.0, 'SCANNUM': 6, 'SCAN': 6,
    # 'DISKSTAT': 'waiting', 'NETSTAT': 'receiving', 'PKTIDX': 26863616, 'DROPAVG': 1.80981e-214, 'DROPTOT': 0,
    # 'DROPBLK': 0, 'STT_IMJD': 57396, 'STT_SMJD': 74701, 'STTVALID': 1, 'NETBUFST': '1/24',
    # 'SCANREM': 6.0, 'STT_OFFS': 0, 'PKTSIZE': 8192, 'NPKT': 16144, 'NDROP': 0}
    # ctr_freq_offset_mhz = focus_freq_min_mhz - lowest_obs_freq_mhz
    # print(f"{focus_freq_min_mhz} - {lowest_obs_freq_mhz} = ctr_freq_offset_mhz: {ctr_freq_offset_mhz}")
    # low_idx = int(np.floor(ctr_freq_offset_mhz / chan_bw_mhz))

    # POL_TYPE note:
    # "String to identify nature of polarisation products.
    # Normally 'AABBCRCI' for NPOL=4 coherence data,
    # where AA and BB are the direct products of the two input channels A and B,
    # and CR and CI are the real and imaginary parts of the cross product A* B.
    test_polarity = first_header.get('POL_TYPE')
    if test_polarity is not None:
        assert test_polarity == 'AABBCRCI'
    # For 8 bit samples, each complex sample consists of two bytes:
    # one byte for real followed by one byte for imaginary.

    # After the header, there is a block of binary data containing several samples
    # from each of the output channels of the polyphase filterbank.
    # They are stored as an array ordered by channel, time, and polarization.
    # block_data format should be
    # (n_chan, n_samples, n_pol):

    # The arrangement of the samples within the data section
    # of a dual polarization observation is as follows:
    #
    # C0T0P0, C0T0P1, C0T1P0, C0T1P1, C0T2P0, C0T2P1, ... C0TtP0, C0TtP1,
    # C1T0P0, C1T0P1, C1T1P0, C1T1P1, C1T2P0, C1T2P1, ... C1TtP0, C1TtP1,
    # ...
    # CcT0P0, CcT0P1, CcT1P0, CcT1P1, CcT2P0, CcT2P1, ... CcTtP0, CcTtP1
    # ...where C0T0P0 represents a complex voltage sample for frequency
    # channel index 0,
    # time sample 0,
    # polarization 0;
    # c is NCHAN-1;
    # and t is NTIME-1.
    # Note that NTIME is usually not present in the header, but can be calculated as:
    # NTIME = BLOCSIZE * 8 / (2 * NPOL * NCHAN * NBITS)

    # reset the file index before we process all the relevant blocks
    r.reset_index()

    test_chan_idx = int(n_coarse_chans // 4)
    print(f"Processing channel {test_chan_idx} (of {n_coarse_chans} ) over n_data_blocks: {n_data_blocks}")
    n_freq_bins = 64 #TODO calculate a reasonable value
    focus_chan_psds, focus_chan_psd_diffs = process_chan_over_all_blocks(r, test_chan_idx, n_data_blocks, n_freq_bins)
    max_psd_diff = np.max(focus_chan_psd_diffs)
    min_psd_diff = np.min(focus_chan_psd_diffs)
    print(f"psd_diff min: {min_psd_diff} max: {max_psd_diff}")
    # diff_scale = max([np.abs(max_psd_diff), np.abs(min_psd_diff)])
    # focus_chan_psd_diffs /= diff_scale # normalize


    # decimate the huge number of coarse chan samples per block to something manageable
    # down_buckets = 1024
    # print(f"decimating {focus_chan_psd_diffs.shape[1]} to {down_buckets}")
    # dec_diffs = np.array([gaussian_decimate(row, down_buckets) for row in focus_chan_psd_diffs])

    # use log scale for plot
    print(f"log scale...")
    perf_start = perf_counter()
    log_psd_diffs = safe_scale_log(focus_chan_psd_diffs)
    log_psds = safe_scale_log(focus_chan_psds)
    print(f"elapsed: {perf_counter()  - perf_start:0.3f} seconds")
    # print(f"log_diffs: {log_psd_diffs}")

    dilation_dim = int(n_freq_bins // 8)
    dilation_size = (2*dilation_dim, dilation_dim)
    # dilated_ascents = ndimage.maximum_filter(log_psd_diffs, size=dilation_size)
    # dilated_psds = ndimage.maximum_filter(log_psds, size=dilation_size)
    # dilated_descents = ndimage.minimum_filter(log_psd_diffs, size=dilation_size)
    dilated_diff_mag = ndimage.maximum_filter(np.abs(log_psd_diffs), size=dilation_size )
    # dilated_diffs = np.add(dilated_ascents, np.abs(dilated_descents))
    # dilated_diffs = dilate_extrema(log_psd_diffs)
    # plot_data = dilated_descents
    # dilated_filtered_psds = ndimage.maximum_filter(ndimage.gaussian_filter(log_psds,sigma=3), size=dilation_size )
    dilated_filtered_psds = ndimage.maximum_filter(ndimage.sobel(np.abs(log_psd_diffs)), size=dilation_size )
    # dilated_filtered_psds = ndimage.laplace(log_psds)

    display_file_name = os.path.basename(data_file_name)
    while True:
        full_screen_dims = (16, 10)
        # fig = plt.figure(figsize=full_screen_dims, constrained_layout=True)
        fig, axes = plt.subplots(nrows=2, figsize=full_screen_dims, constrained_layout=True, sharex=True,sharey=True)
        # fig.subplots_adjust(hspace=0)
        fig.suptitle(f"chan {test_chan_idx} PSD diffs| {display_file_name}")
        # img0 = plt.imshow(plot_data.T, aspect='auto', cmap='inferno')
        img0 = axes[0].imshow(dilated_filtered_psds.T, aspect='auto', cmap='viridis')
        # axes[0].set_ylabel('Freq bin')
        img1 = axes[1].imshow(dilated_diff_mag.T, aspect='auto', cmap='inferno')
        # axes[1].set_ylabel('Freq bin')
        axes[1].set_xlabel('Time')
        cbar0 = fig.colorbar(img0, ax=axes[0])
        cbar0.set_label('ΔPSD dB', rotation=270, labelpad=15)
        cbar0.ax.yaxis.set_ticks_position( 'left')

        cbar1 = fig.colorbar(img1, ax=axes[1])
        cbar1.set_label('abs(ΔPSD) dB', rotation=270, labelpad=15)
        cbar1.ax.yaxis.set_ticks_position( 'left')

        img_save_path = f"./plots/pow_diffs/{display_file_name}.N{n_freq_bins}_sflux.png"
        print(f"saving image to:\n{img_save_path} ...")
        plt.savefig(img_save_path)

        plt.show()
        return


if __name__ == "__main__":
    main()
