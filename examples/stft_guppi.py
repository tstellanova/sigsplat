"""
Process a raw GUPPI file
Look for changes in power over time,
using Short Time Fourier Transform (STFT) methods
"""
import argparse

import blimpy
from blimpy import GuppiRaw
import matplotlib
import matplotlib.pyplot as plt
import sys

import numpy as np
import scipy
import os

from blimpy.guppi import EndOfFileError
from scipy import ndimage
from scipy.optimize import dual_annealing

from time import perf_counter
from scipy.signal import ShortTimeFFT

matplotlib.use('qtagg')


MAX_PLT_POINTS = 65536 * 4  # Max number of points in matplotlib plot


def skip_to_block(bb_reader, dest_block_num=2):
    """ Quickly seek through the file to skip to a block of interest

    Returns:
        n_blocks (int): Number of blocks skipped
    """
    bb_reader.file_obj.seek(0)
    n_blocks = 0
    limit_block_num = dest_block_num - 1
    while n_blocks < limit_block_num:
        try:
            header, data_idx = bb_reader.read_header()
            bb_reader.file_obj.seek(data_idx)
            bloc_size = header['BLOCSIZE']
            # print(f"bloc_size: {bloc_size}")
            bb_reader.file_obj.seek(bloc_size, 1)
            n_blocks += 1
        except EndOfFileError:
            break

    return n_blocks


def process_dual_pol_block(stft_obj=None, full_energy_correction=None, chan_pol_a=None, chan_pol_b=None, n_freq_bins=1024, fs=100e6):
    perf_start = perf_counter()
    if chan_pol_a is None:
        chan_pol_a = []
    if chan_pol_b is None:
        chan_pol_b = []

    # TODO alt: squeeze?
    chan_pol_a_view = chan_pol_a.astype(np.float32).view('complex64')[..., 0]
    chan_pol_b_view = chan_pol_b.astype(np.float32).view('complex64')[..., 0]

    # assume that signals of interest will be correlated in A and B polarities
    # correlated_signal = np.multiply(chan_pol_a_view, np.conjugate(chan_pol_b_view))

    # window = scipy.signal.get_window('hann', n_freq_bins)  # Hanning window
    # energy_correction = 1 / np.sum(window ** 2)
    # full_correction = (2 / (n_freq_bins * fs)) / energy_correction
    # stft_obj = ShortTimeFFT(fs=fs, win=window, hop=n_freq_bins // 2, fft_mode='centered')

    zxx_pol_a = stft_obj.stft(chan_pol_a_view)
    # pol_a_psd = (np.abs(zxx_pol_a) ** 2)/energy_correction # for testing against single-block plot
    # pol_a_psd = (2 / (n_freq_bins * fs)) * (np.abs(zxx_pol_a) ** 2) / energy_correction  # PSD for a one-sided spectrum
    pol_a_psd = full_energy_correction * (np.abs(zxx_pol_a) ** 2)
    block_psd = pol_a_psd

    zxx_pol_b = stft_obj.stft(chan_pol_b_view)
    # pol_b_psd = (np.abs(zxx_pol_b) ** 2)/ energy_correction# for testing against single-block plot
    # pol_b_psd = (2 / (n_freq_bins * fs)) * (np.abs(zxx_pol_b) ** 2) / energy_correction  # PSD for a one-sided spectrum
    pol_b_psd = full_energy_correction * (np.abs(zxx_pol_b) ** 2)
    block_psd += pol_b_psd # TODO use correlation instead? these are two orthogonal polarizations

    # print(f"Dual PSD calc >>> elapsed: {perf_counter() - perf_start:0.3f} seconds")

    return block_psd


def process_all_channels(data_reader:GuppiRaw=None,
                         plots_path=None,
                         n_coarse_chans=1,
                         n_freq_bins=256,
                         n_data_blocks=4,
                         sampling_rate_hz=100E6,
                         start_freq_mhz=1e3,
                         chan_bw_delta_mhz=1e6,
                         display_file_name=''):

    for chan_idx in range(n_coarse_chans):
        # we read one block after another from the input file:
        # we collect spectral power calculated for each channel along the way:
        # this means we must reset the file reader to the first block, for each new channel
        data_reader.reset_index()

        chan_start_mhz = start_freq_mhz + chan_idx*chan_bw_delta_mhz
        chan_title = f"chan {chan_idx:03d} | {chan_start_mhz:0.3f} MHz | {display_file_name}"
        print(f"Processing channel {chan_idx+1}/{n_coarse_chans} over {n_data_blocks} blocks")
        perf_start_chan_process = perf_counter()
        # TODO flip this and process all blocks in order, processing every channel at each block iteration
        focus_chan_psds, focus_chan_psd_diffs = process_chan_over_all_blocks(
            data_reader, chan_idx, n_data_blocks, n_freq_bins, sampling_rate_hz)

        print(f">>> chan psds: {np.min(focus_chan_psds):0.3e} ... {np.max(focus_chan_psds):0.3e}")
        print(f">>> chan psd_diff: {np.min(focus_chan_psd_diffs):0.3e} ... {np.max(focus_chan_psd_diffs):0.3e}")
        print(f"Processing chan {chan_idx+1} >>> elapsed: {perf_counter() - perf_start_chan_process:0.3f} seconds")

        # use log scale for plot
        print(f"log scale...")
        perf_start_log_scale = perf_counter()
        log_psd_diffs = safe_scale_log(focus_chan_psd_diffs)
        # normalized_psds = normalize_data(focus_chan_psds)
        log_psds = safe_scale_log(focus_chan_psds)
        print(f"Log scaling >>> elapsed: {perf_counter() - perf_start_log_scale:0.3f} seconds")

        perf_start_vis_filt = perf_counter()
        dilation_dim = int(n_freq_bins // 10)
        filtered_psds = ndimage.gaussian_filter(log_psds,sigma=1)
        dilated_diff_max = ndimage.maximum_filter(log_psd_diffs, size=(2*dilation_dim, dilation_dim))
        print(f"Visual filtering >>> elapsed: {perf_counter() - perf_start_vis_filt:0.3f} seconds")

        perf_start_plot_psds = perf_counter()
        full_screen_dims = (16, 10)
        fig, axes = plt.subplots(nrows=2, figsize=full_screen_dims, constrained_layout=True, sharex=True, sharey=True)
        fig.suptitle(chan_title)
        img0 = axes[0].imshow(filtered_psds, aspect='auto', cmap='inferno') #, vmin=9, vmax=16)
        img1 = axes[1].imshow(dilated_diff_max, aspect='auto', cmap='viridis') #, vmin=15, vmax=80)
        axes[1].set_xlabel('Time')
        cbar0 = fig.colorbar(img0, ax=axes[0])
        cbar0.set_label('PSD dB', rotation=270, labelpad=15)
        cbar0.ax.yaxis.set_ticks_position('left')

        cbar1 = fig.colorbar(img1, ax=axes[1])
        cbar1.set_label('mag(ΔPSD) dB', rotation=270, labelpad=15)
        cbar1.ax.yaxis.set_ticks_position('left')

        print(f"Plotting >>> elapsed: {perf_counter() - perf_start_plot_psds:0.3f} seconds")

        if plots_path is not None:
            img_save_path = f"{plots_path}pow_diffs/{display_file_name}.ch{chan_idx:d}_B{n_data_blocks:d}_N{n_freq_bins}_sflux.png"
            print(f"saving plot to:\n{img_save_path} ...")
            perf_start = perf_counter()
            plt.savefig(img_save_path)
            print(f"Saving img file >>> elapsed: {perf_counter() - perf_start:0.3f} seconds")

def plot_spectrum_dual_pol(raw_reader:GuppiRaw,
                           display_file_name=None, out_filepath=None,
                           start_freq_mhz:float=2e3, stop_freq_mhz:float=1e3, center_freq_mhz:float=1.5e3,
                           flag_show=False):
    """
    Plot simple magnitude of FFT for two polarities of input data
    """
    unused_header, pol_x, pol_y = raw_reader.read_next_data_block_int8()
    print(f"pol_x: {pol_x.shape} {pol_x.dtype}") # (64, 524288, 2)

    pol_x_complex = pol_x.astype(np.float32).view('complex64')[..., 0]
    pol_y_complex = pol_y.astype(np.float32).view('complex64')[..., 0]
    print(f"pol_x_complex: {pol_x_complex.shape} {pol_x_complex.dtype}")

    print("Computing dual pol FFT...")
    pol_x_fft_mag = np.abs(np.fft.fft(pol_x_complex))
    pol_y_fft_mag = np.abs(np.fft.fft(pol_y_complex))
    print(f"pol_x_fft_mag: {pol_x_fft_mag.shape} {pol_x_fft_mag.dtype}")

    pol_x_fft_mag = pol_x_fft_mag.flatten()
    pol_y_fft_mag = pol_y_fft_mag.flatten()
    print(f"pol_x_fft_mag flat: {pol_x_fft_mag.shape} {pol_x_fft_mag.dtype}")

    # TODO might be faster / better to use gaussian decimation?
    # Rebin to max number of points
    if pol_x_fft_mag.shape[0] > MAX_PLT_POINTS:
        dec_fac = int(pol_x_fft_mag.shape[0] / MAX_PLT_POINTS)
        print(f"reshaping fac: {dec_fac} {pol_x_fft_mag.dtype}")
        pol_x_fft_mag = blimpy.utils.rebin(pol_x_fft_mag, dec_fac)
        pol_y_fft_mag = blimpy.utils.rebin(pol_y_fft_mag, dec_fac)

    print(f"pol_x min: {np.min(pol_x)} max: {np.max(pol_x)}")
    print(f"pol_y min: {np.min(pol_y)} max: {np.max(pol_y)}")

    print(f"Plotting...{pol_x_fft_mag.shape} {pol_x_fft_mag.dtype}")
    plt.ylabel("mag(FFT) [dB]")
    plt.plot(10 * np.log10(pol_x_fft_mag), label='pol_x')
    plt.plot(10 * np.log10(pol_y_fft_mag), label='pol_y')

    if display_file_name is not None:
        plt.title(f"Spectrum: {display_file_name}")

    orig_xticks = plt.gca().get_xticks()
    # print(f"orig_xticks: {orig_xticks}")
    ticks_len = len(orig_xticks)
    coarse_freqs = np.linspace(start_freq_mhz, stop_freq_mhz, ticks_len-2)
    freq_step = coarse_freqs[1] - coarse_freqs[0]
    expanded_coarse_freqs = np.concatenate(([coarse_freqs[0] - freq_step], coarse_freqs, [coarse_freqs[-1] + freq_step]))
    # print(f"freqs: {expanded_coarse_freqs.shape} range: {expanded_coarse_freqs[0]} ... {expanded_coarse_freqs[-1]}" )
    plt.xticks(ticks=orig_xticks, labels=expanded_coarse_freqs)

    xlim_range = plt.xlim()
    # print(f"xlim_range {xlim_range}")
    freq_ratio = float(xlim_range[1] - xlim_range[0])/(expanded_coarse_freqs[-1] - expanded_coarse_freqs[0])
    scaled_freq_center = (center_freq_mhz - expanded_coarse_freqs[0])*freq_ratio + xlim_range[0]
    # print(f"freq_ratio {freq_ratio}  scaled_freq_center {scaled_freq_center}")
    # place line at observation frequency center
    plt.axvline(x=scaled_freq_center, color='r')
    plt.legend()

    if out_filepath is not None:
        print(f"saving spectrum to:\n{out_filepath} ...")
        plt.savefig(out_filepath)
    if flag_show:
        plt.show()


def process_chan_over_all_blocks(raw_reader:GuppiRaw=None, chan_idx:int=0, n_blocks:int=1, n_freq_bins:int=1024, fs=100E6):
    # focus_chan_psd_diffs = []
    all_chan_psds = None
    # prior_psd = None
    samples_per_chan_per_block = None
    print(f"Collecting chan {chan_idx+1} PSDs over n_blocks: {n_blocks}")
    chan_psds_perf_start = perf_counter()
    block_range = range(n_blocks)

    # prepare reusable STFT objects
    window = scipy.signal.get_window('hann', n_freq_bins)  # Hanning window
    energy_correction = 1 / np.sum(window ** 2)
    full_energy_correction = (2 / (n_freq_bins * fs)) / energy_correction
    stft_obj = ShortTimeFFT(fs=fs, win=window, hop=n_freq_bins // 2, fft_mode='centered')

    for block_idx in block_range:
        # print(f"Processing chan {chan_idx} block: {block_idx}/{n_blocks}")
        perf_start_blockchan_process = perf_counter()
        perf_start_read_block = perf_start_blockchan_process
        block_hdr, data_pol_a, data_pol_b = raw_reader.read_next_data_block_int8()
        # print(f"block_hdr: {block_hdr}")
        if block_hdr is None:
            print(f"block_hdr: {block_hdr}")
            break
        # print(f"Read block {block_idx+1} >>> elapsed: {perf_counter() - perf_start_read_block:0.3f} seconds")

        chan_pol_a = data_pol_a[chan_idx]
        chan_pol_b = data_pol_b[chan_idx]

        if samples_per_chan_per_block is None:
            samples_per_chan_per_block = chan_pol_a.shape[0]
            print(f"data_pol_a: {data_pol_a.shape} {data_pol_a.dtype}")
            print(f"chan_pol_a : {chan_pol_a.shape} samples_per_chan_per_block: {samples_per_chan_per_block}")

        block_psd = process_dual_pol_block(stft_obj, full_energy_correction, chan_pol_a, chan_pol_b, n_freq_bins, fs)
        # print(f"block_psd {block_psd.shape}") # something like (n_freq_bins, samples_per_chan_per_block)

        if all_chan_psds is None:
            n_blocks_processing = block_range.stop - block_range.start
            print(f"n_blocks_processing: {n_blocks_processing}")
            extent_all_psds = n_blocks_processing * block_psd.shape[1]
            full_extent_shape = (block_psd.shape[0], extent_all_psds)
            print(f"full_extent_shape: {full_extent_shape}")
            all_chan_psds = np.zeros(full_extent_shape, dtype=np.float32)

        # first, collect all the PSDs into the PSD result block
        block_inset = block_idx * block_psd.shape[1]
        # print(f"block_inset {block_inset}")
        all_chan_psds[0:block_psd.shape[0], block_inset:block_inset+block_psd.shape[1]] = block_psd
        print(f"Chan {chan_idx+1} Block {block_idx+1} / {n_blocks} PDSs >>> elapsed: {perf_counter() - perf_start_blockchan_process:0.3f} seconds")
    print(f"Collecting chan {chan_idx+1} PDSs >>> elapsed: {perf_counter() - chan_psds_perf_start:0.3f} seconds")

    # print(f"init chan psds: {np.min(all_chan_psds):0.3e} ... {np.max(all_chan_psds):0.3e}")

    print(f"Calculate all PSD diffs {all_chan_psds.shape} ...")
    psd_diffs_perf_start = perf_counter()
    all_chan_psd_diffs = np.zeros_like(all_chan_psds)
    all_chan_psd_diffs[:, 1:] = all_chan_psds[:, 1:] - all_chan_psds[:, :-1]
    print(f"Chan PSD {chan_idx+1} diffs >>> elapsed: {perf_counter() - psd_diffs_perf_start:0.3f} seconds")

    # print(f"all_chan_psd_diffs {all_chan_psd_diffs.shape}")
    print(f"init chan psd_diffs: {np.min(all_chan_psd_diffs):0.3e} ... {np.max(all_chan_psd_diffs):0.3e}")

    return all_chan_psds, all_chan_psd_diffs


def main():
    parser = argparse.ArgumentParser(description='Analyze power over time in GUPPI file')
    parser.add_argument('src_path', nargs='?',
                        help="Source raw file path eg 'blc21_guppi.raw' with the '.raw' file extension",
                        default="../../baseband/bloda/"
                        # "guppi_58198_27514_685364_J0835-4510_B3_0001.0000.raw"
                                "blc00_guppi_57598_81738_Diag_psr_j1903+0327_0019.0005.raw"
                                # "blc4_2bit_guppi_57432_24865_PSR_J1136+1551_1406_025_0002.0001.raw"
                        )
    parser.add_argument('-o', dest='plots_path', type=str, default='./plots/',
                        help='Output directory for plots')

    args = parser.parse_args()
    data_file_name = args.src_path
    plots_path = args.plots_path
    print(f"Processing: {data_file_name} , plots to: {plots_path}")

    display_file_name = os.path.splitext(os.path.basename(data_file_name))[0]

    bb_reader = GuppiRaw(data_file_name)
    # Seek through the file to find how many data blocks there are in the file
    # max_blocks = int(4)
    n_data_blocks = bb_reader.find_n_data_blocks()
    print(f"total n_data_blocks: {n_data_blocks}")

    # reads the first header and then seek to file offset zero
    first_header = bb_reader.read_first_header()
    print(f"first_header: {first_header}")

    n_coarse_chans = int(first_header['OBSNCHAN'])

    # Per davidm, 'CHAN_BW': 2.9296875 (MHz) and 'TBIN': 3.41333333333e-07 should be inverses of each other,
    # and they usually are, approximately, unless CHAN_BW is not present
    time_bin = float(first_header['TBIN'])
    print(f"TBIN: {time_bin:0.3e}")
    sampling_rate_hz = (1 / time_bin)

    # OBSBW: [MHz] width of passband; negative sign indicates spectral flip
    total_obs_bw_mhz = float(first_header['OBSBW'])
    print(f"total_obs_bw_mhz: {total_obs_bw_mhz:0.3e}")

    # CHAN_BW [MSPS] sample rate for a channel.  Negative sign indicates spectral flip
    chan_bw_delta_mhz = first_header.get('CHAN_BW')
    if chan_bw_delta_mhz is not None:
        chan_bw_delta_mhz = float(chan_bw_delta_mhz)
        chan_bw_sps = chan_bw_delta_mhz * 1E6
    else:
        chan_bw_sps = sampling_rate_hz
        chan_bw_delta_mhz = chan_bw_sps / 1e6
    print(f"chan_bw_sps: {chan_bw_sps} sampling_rate_hz: {sampling_rate_hz}")

    # OBSFREQ: [MHz] center of the RF passband
    center_obs_freq_mhz = float(first_header['OBSFREQ'])
    print(f"chan_bw_delta_mhz: {chan_bw_delta_mhz}")

    half_n_chans = n_coarse_chans // 2
    half_obs_bw_mhz = half_n_chans*np.abs(chan_bw_delta_mhz)
    if chan_bw_delta_mhz < 0:
        start_freq_mhz = center_obs_freq_mhz + half_obs_bw_mhz
        stop_freq_mhz = center_obs_freq_mhz - half_obs_bw_mhz
    else:
        start_freq_mhz = center_obs_freq_mhz - half_obs_bw_mhz
        stop_freq_mhz = center_obs_freq_mhz + half_obs_bw_mhz

    print(f"range: {start_freq_mhz} - {stop_freq_mhz} MHz, center: {center_obs_freq_mhz} MHz, total bw: {total_obs_bw_mhz} MHz")

    n_pols = int(first_header['NPOL'])
    n_bits = int(first_header['NBITS'])
    block_size = int(first_header['BLOCSIZE'])
    # NTIME = BLOCSIZE * 8 / (2 * NPOL * NCHAN * NBITS)
    ntime_calc = block_size * 8 / (2 * n_pols * n_coarse_chans * n_bits)
    print(f"NTIME: {ntime_calc} BLOCSIZE: {block_size} NPOL: {n_pols} NCHAN: {n_coarse_chans} NBITS: {n_bits}")

    # NDIM is the number of samples per channel in the block; i.e.,
    # BLOCSIZE/(OBSNCHAN*NPOL*(NBITS/8))
    samples_per_chan_per_block = int((block_size*8) / (n_pols * n_coarse_chans * n_bits))
    print(f"n_dim (samples_per_chan_per_block): {samples_per_chan_per_block}")

    # if n_pols > 2:
    #     n_pols = np.sqrt(n_pols)

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
    # low_idx = int(np.floor(ctr_freq_offset_mhz / chan_bw_msps))

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

    # reset the file  before we process the signature full-spectrum block
    bb_reader.reset_index()
    # forward through the data blocks til we're halfway through the input
    sample_block_num = int(n_data_blocks // 2)
    sample_block_idx =  sample_block_num - 1
    stop_block_num = sample_block_idx # we skip all these blocks
    print(f"Forwarding file to block {sample_block_num}")
    n_blocks_skipped = skip_to_block(bb_reader, sample_block_num)
    # for i in range(stop_block_num):
    #     # TODO add method to blimpy, using a faster file seek method to skip to the desired block
    #     blk_hdr, blk_data_x, blk_data_y = bb_reader.read_next_data_block_int8()
    #     if i == (stop_block_num - 1):
    #         print(f"skipped to block num {stop_block_num} like {blk_data_x.shape} dtype: {blk_data_x.dtype}")
    print(f"skipped {n_blocks_skipped} blocks to sample block {sample_block_num}")

    full_screen_dims = (16, 10)
    plt.figure(figsize=full_screen_dims, constrained_layout=True)
    img_save_path = f"{plots_path}spectra/{display_file_name}_bl_dualmag.png"
    plot_spectrum_dual_pol(bb_reader,
                           display_file_name=display_file_name,
                           out_filepath=img_save_path,
                           start_freq_mhz=start_freq_mhz,
                           stop_freq_mhz=stop_freq_mhz,
                           center_freq_mhz=center_obs_freq_mhz,
                           flag_show=False
                           )

    # reload the data file for processing all of the channels, clearing out any reader state
    bb_reader = GuppiRaw(data_file_name)
    n_freq_bins = 128  # TODO calculate a reasonable value of bins
    process_all_channels(bb_reader, n_coarse_chans=n_coarse_chans,
                         plots_path=plots_path,
                         n_freq_bins=n_freq_bins,
                         n_data_blocks=n_data_blocks,
                         sampling_rate_hz=sampling_rate_hz,
                         start_freq_mhz=start_freq_mhz,
                         chan_bw_delta_mhz=chan_bw_delta_mhz,
                         display_file_name=display_file_name,
                         )



if __name__ == "__main__":
    main()
