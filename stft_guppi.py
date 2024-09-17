"""
Process a raw GUPPI file
Look for changes in power over time
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

from scipy import ndimage
from scipy.optimize import dual_annealing

from time import perf_counter
from scipy.signal import ShortTimeFFT

matplotlib.use('qtagg')


def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)


def safe_scale_log(data):
    return 10 * np.sign(data) * np.log10(np.abs(data) + sys.float_info.epsilon)


def gaussian_decimate(data, num_out_buckets, sigma=1.0):
    # Step 1: Apply Gaussian filter
    filtered_data = ndimage.gaussian_filter1d(data, sigma=sigma)

    # Step 2: Decimate (downsample)
    decimation_factor = len(data) // num_out_buckets
    decimated_data = filtered_data[::decimation_factor]

    return decimated_data


def process_dual_pol_block(chan_pol_a=None, chan_pol_b=None, n_freq_bins=1024,  fs=100e6):
    if chan_pol_a is None:
        chan_pol_a = []
    if chan_pol_b is None:
        chan_pol_b = []

    # # For 8 bit samples, each complex sample consists of two bytes:
    # # one byte for real followed by one byte for imaginary.
    # chan_pol_a_iq = chan_pol_a.f
    # ()
    # chan_pol_b_iq = chan_pol_b.flatten()

    # TODO alt: squeeze?
    chan_pol_a_view = chan_pol_a.astype(np.float32).view('complex64')[..., 0]
    # print(f"chan_pol_a_view initial: {chan_pol_a_view.shape}")
    # print(f"chan_pol_a_view final: {chan_pol_a_view.shape}")
    chan_pol_b_view = chan_pol_b.astype(np.float32).view('complex64')[..., 0]

    # chan_pol_a_view = chan_pol_a_iq.astype(np.float32).view('complex64')
    # # print(f"chan_pol_a_view {chan_pol_a_view.shape}")
    # chan_pol_b_view = chan_pol_b_iq.astype(np.float32).view('complex64')

    # assume that signals of interest will be correlated in A and B polarities
    # correlated_signal = np.multiply(chan_pol_a_view, np.conjugate(chan_pol_b_view))

    window = scipy.signal.get_window('hann', n_freq_bins)  # Hanning window
    energy_correction = 1 / np.sum(window ** 2)
    stft_obj = ShortTimeFFT(fs=fs, win=window, hop=n_freq_bins // 2, fft_mode='centered')

    # zxx_all = stft_obj.stft(correlated_signal)
    # block_psd = (2 / (n_freq_bins * fs)) * (np.abs(zxx_all) ** 2) / energy_correction  # PSD for a one-sided spectrum

    zxx_pol_a = stft_obj.stft(chan_pol_a_view)
    pol_a_psd = (2 / (n_freq_bins * fs)) * (np.abs(zxx_pol_a) ** 2) / energy_correction  # PSD for a one-sided spectrum

    zxx_pol_b = stft_obj.stft(chan_pol_b_view)
    pol_b_psd = (2 / (n_freq_bins * fs)) * (np.abs(zxx_pol_b) ** 2) / energy_correction  # PSD for a one-sided spectrum

    block_psd = pol_a_psd + pol_b_psd # TODO use correlation instead?

    return block_psd


def process_all_channels(data_reader:GuppiRaw=None,
                         n_coarse_chans=1,
                         n_freq_bins=256,
                         n_data_blocks=4,
                         sampling_rate_hz=100E6,
                         start_freq_mhz=1e3,
                         chan_bw_delta_mhz=1e6,
                         display_file_name=''):

    # TODO temporary -- scan a few chans around the center of the overall observation
    mid_chan_idx = n_coarse_chans // 2
    chan_range = range(mid_chan_idx - 2, mid_chan_idx + 2)
    for chan_idx in chan_range:
        # we read one block after another from the input file:
        # we collect spectral power calculated for each channel along the way:
        # this means we just reset the file reader to the first block, for each new channel
        data_reader.reset_index()

        chan_start_mhz = start_freq_mhz + chan_idx*chan_bw_delta_mhz
        chan_title = f"chan {chan_idx:03d} | {chan_start_mhz:0.3f} MHz | {display_file_name}"
        print(f"Processing channel {chan_idx} (of {n_coarse_chans} ) over n_data_blocks: {n_data_blocks}")
        focus_chan_psds, focus_chan_psd_diffs = process_chan_over_all_blocks(data_reader, chan_idx, n_data_blocks, n_freq_bins,
                                                                             sampling_rate_hz)
        min_psd = np.min(focus_chan_psds)
        max_psd = np.max(focus_chan_psds)
        psd_range = max_psd - min_psd
        print(f">>> chan min_psd: {min_psd} psd_range: {psd_range:0.3e}")

        min_psd_diff = np.min(focus_chan_psd_diffs)
        max_psd_diff = np.max(focus_chan_psd_diffs)
        psd_diff_range = max_psd_diff - min_psd_diff
        print(f">>> min_psd_diff: {min_psd_diff} psd_diff_range: {psd_diff_range:0.3e}")

        # use log scale for plot
        print(f"log scale...")
        perf_start = perf_counter()
        log_psd_diffs = safe_scale_log(focus_chan_psd_diffs)
        normalized_psds = normalize_data(focus_chan_psds)
        log_psds = safe_scale_log(normalized_psds)
        print(f"elapsed: {perf_counter() - perf_start:0.3f} seconds")

        dilation_dim = int(n_freq_bins // 10)
        # dilated_psds = ndimage.maximum_filter(log_psds, size=(dilation_dim, dilation_dim))
        # dilated_diff_mag = ndimage.maximum_filter(np.abs(log_psd_diffs), size=dilation_size )
        dilated_diff_max = ndimage.maximum_filter(log_psd_diffs, size=(2*dilation_dim, dilation_dim))

        # dilated_descents = ndimage.minimum_filter(log_psd_diffs, size=dilation_size)
        # dilated_diffs = np.add(dilated_ascents, np.abs(dilated_descents))
        # dilated_diffs = dilate_extrema(log_psd_diffs)
        # plot_data = dilated_descents
        # dilated_filtered_psds = ndimage.maximum_filter(ndimage.gaussian_filter(log_psds,sigma=3), size=dilation_size )
        # dilated_filtered_psds = ndimage.maximum_filter(ndimage.sobel(np.abs(log_psd_diffs)), size=dilation_size )
        # dilated_filtered_psds = ndimage.laplace(log_psds)

        full_screen_dims = (16, 10)
        fig, axes = plt.subplots(nrows=2, figsize=full_screen_dims, constrained_layout=True, sharex=True, sharey=True)
        fig.suptitle(chan_title)
        img0 = axes[0].imshow(log_psds.T, aspect='auto', cmap='inferno') #, vmin=9, vmax=16)
        img1 = axes[1].imshow(dilated_diff_max.T, aspect='auto', cmap='viridis') #, vmin=15, vmax=80)
        axes[1].set_xlabel('Time')
        cbar0 = fig.colorbar(img0, ax=axes[0])
        cbar0.set_label('PSD dB', rotation=270, labelpad=15)
        cbar0.ax.yaxis.set_ticks_position('left')

        cbar1 = fig.colorbar(img1, ax=axes[1])
        cbar1.set_label('abs(ΔPSD) dB', rotation=270, labelpad=15)
        cbar1.ax.yaxis.set_ticks_position('left')

        img_save_path = f"./img/pow_diffs/{display_file_name}.ch{chan_idx:d}_B{n_data_blocks:d}_N{n_freq_bins}_sflux.png"
        print(f"saving image to:\n{img_save_path} ...")
        perf_start = perf_counter()
        plt.savefig(img_save_path)
        print(f"elapsed: {perf_counter() - perf_start:0.3f} seconds")

        plt.show()



def plot_spectrum_dual_pol(raw_reader:GuppiRaw,
                           display_file_name=None, out_filepath=None,
                           start_freq_mhz=2e3, stop_freq_mhz=1e3, center_freq_mhz=1.5e3,
                           plot_db=True,
                           flag_show=False):
    """ Do a (slow) numpy FFT and take power of data
        Plot simple magnitude of FFT for input data
    """
    MAX_PLT_POINTS = 65536 * 4  # Max number of points in matplotlib plot
    unused_header, pol_x, pol_y = raw_reader.read_next_data_block_int8()
    print(f"pol_x: {pol_x.shape} {pol_x.dtype}") # (64, 524288, 2)

    pol_x_complex = pol_x.astype(np.float32).view('complex64')[..., 0]
    pol_y_complex = pol_y.astype(np.float32).view('complex64')[..., 0]
    print(f"pol_x_complex: {pol_x_complex.shape} {pol_x_complex.dtype}")

    print("Computing FFT...")
    pol_x_fft_mag = np.abs(np.fft.fft(pol_x_complex)).flatten()
    pol_y_fft_mag = np.abs(np.fft.fft(pol_y_complex)).flatten()

    # Rebin to max number of points
    if pol_x_fft_mag.shape[0] > MAX_PLT_POINTS:
        dec_fac = int(pol_x_fft_mag.shape[0] / MAX_PLT_POINTS)
        print(f"reshaping fac: {dec_fac} {pol_x_fft_mag.dtype}")
        pol_x_fft_mag = blimpy.utils.rebin(pol_x_fft_mag, dec_fac)
        pol_y_fft_mag = blimpy.utils.rebin(pol_y_fft_mag, dec_fac)

    print(f"Plotting...{pol_x_fft_mag.shape} {pol_x_fft_mag.dtype}")
    if plot_db:
        plt.ylabel("mag(FFT) [dB]")
        plt.plot(10 * np.log10(pol_x_fft_mag), label='pol_x')
        plt.plot(10 * np.log10(pol_y_fft_mag), label='pol_y')
    else:
        plt.ylabel("mag(FFT)")
        plt.plot(pol_x_fft_mag, label='pol_x')
        plt.plot(pol_y_fft_mag, label='pol_y')

    if display_file_name is not None:
        plt.title(f"Spectrum: {display_file_name}")

    orig_xticks = plt.gca().get_xticks()
    print(f"orig_xticks: {orig_xticks}")
    ticks_len = len(orig_xticks)
    coarse_freqs = np.linspace(start_freq_mhz, stop_freq_mhz, ticks_len-2)
    freq_step = coarse_freqs[1] - coarse_freqs[0]
    expanded_coarse_freqs = np.concatenate(([coarse_freqs[0] - freq_step], coarse_freqs, [coarse_freqs[-1] + freq_step]))
    print(f"freqs: {expanded_coarse_freqs.shape} range: {expanded_coarse_freqs[0]} ... {expanded_coarse_freqs[-1]}" )
    plt.xticks(ticks=orig_xticks, labels=expanded_coarse_freqs)

    xlim_range = plt.xlim()
    # print(f"xlim_range {xlim_range}")
    freq_ratio = float(xlim_range[1] - xlim_range[0])/(expanded_coarse_freqs[-1] - expanded_coarse_freqs[0])
    scaled_freq_center = (center_freq_mhz - expanded_coarse_freqs[0])*freq_ratio + xlim_range[0]
    # print(f"freq_ratio {freq_ratio}  scaled_freq_center {scaled_freq_center}")
    plt.axvline(x=scaled_freq_center, color='r')

    plt.legend()
    if out_filepath is not None:
        print(f"saving spectrum to:\n{out_filepath} ...")
        plt.savefig(out_filepath)
    if flag_show:
        plt.show()


def process_chan_over_all_blocks(raw_reader:GuppiRaw=None, chan_idx:int=0, n_blocks:int=1, n_freq_bins:int=1024, fs=100E6):
    focus_chan_psd_diffs = []
    all_focus_chan_psds = []
    prior_psd = None
    samples_per_chan_per_block = None
    # print(f"Processing chan {chan_idx} over n_blocks: {n_blocks}")
    # TODO tmp: use a limited number of blocks
    half_n_blocks = n_blocks // 2
    block_range = range(half_n_blocks - 4, half_n_blocks + 4)
    for block_idx in block_range:
        print(f"Processing chan {chan_idx} block: {block_idx}/{n_blocks}")
        perf_start = perf_counter()
        block_hdr, data_pol_a, data_pol_b = raw_reader.read_next_data_block_int8()
        # print(f"block_hdr: {block_hdr}")
        if block_hdr is None:
            print(f"block_hdr: {block_hdr}")
            break
        print(f"elapsed: {perf_counter() - perf_start:0.3f} seconds")

        chan_pol_a = data_pol_a[chan_idx]
        chan_pol_b = data_pol_b[chan_idx]

        if samples_per_chan_per_block is None:
            samples_per_chan_per_block = chan_pol_a.shape[0]
            print(f"data_pol_a: {data_pol_a.shape} {data_pol_a.dtype}")
            print(f"chan_pol_a : {chan_pol_a.shape} samples_per_chan_per_block: {samples_per_chan_per_block}")


        block_psd = process_dual_pol_block(chan_pol_a, chan_pol_b, n_freq_bins, fs)
        # print(f"block_psd {block_psd.shape}") # something like (n_freq_bins, samples_per_chan_per_block)

        print(f"Calculate block PSD {block_psd.shape} diffs...")
        perf_start = perf_counter()
        for t_idx in range(0, block_psd.shape[1]):
            cur_psd = block_psd[:, t_idx]
            all_focus_chan_psds.append(cur_psd)
            if prior_psd is None:
                # the zeroth psd_diff will always be zero
                prior_psd = cur_psd
                continue
            cur_vps_diff = np.subtract(cur_psd, prior_psd)
            focus_chan_psd_diffs.append(cur_vps_diff)
            prior_psd = cur_psd
        print(f"elapsed: {perf_counter() - perf_start:0.3f} seconds")

    all_chan_psds = np.asarray(all_focus_chan_psds)
    all_chan_psd_diffs = np.asarray(focus_chan_psd_diffs)

    return all_chan_psds, all_chan_psd_diffs


def main():
    parser = argparse.ArgumentParser(description='Analyze GUPPI file')
    parser.add_argument('src_path', nargs='?',
                        help="Source raw file path eg 'blc21_guppi.raw' with the '.raw' file extension",
                        default="../../baseband/bloda/"
                        # "guppi_58198_27514_685364_J0835-4510_B3_0001.0000.raw"
                                "blc00_guppi_57598_81738_Diag_psr_j1903+0327_0019.0005.raw"
                                # "blc4_2bit_guppi_57432_24865_PSR_J1136+1551_1406_025_0002.0001.raw"
                        )
    parser.add_argument('-o', dest='outdir', type=str, default='./data',
                        help='Output directory processed files')

    args = parser.parse_args()
    data_file_name = args.src_path
    out_dir = args.outdir
    print(f"Loading: {data_file_name} , output to: {out_dir}")

    display_file_name = os.path.splitext(os.path.basename(data_file_name))[0]

    bb_reader = GuppiRaw(data_file_name)
    # Seek through the file to find how many data blocks there are in the file
    # max_blocks = int(4)
    n_data_blocks = bb_reader.find_n_data_blocks()
    print(f"total n_data_blocks: {n_data_blocks}")
    # if n_data_blocks > max_blocks:
    #     print(f"clamping to {max_blocks} blocks")
    #     n_data_blocks = max_blocks
    # TODO this is a little confusing, but:
    # sometimes n_data_blocks is, say, 128;
    # if we then read 4 blocks for each channel,
    # we only get 32 channel reads before we run out of memory

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

    # reset the file  before we process all the relevant blocks
    bb_reader.reset_index()
    # # forward through the data blocks til we're halfway there
    # for i in range(n_data_blocks // 2):
    #     blk_hdr, blk_data = bb_reader.read_next_data_block()
    #     if i == 0:
    #         print(f"raw block {blk_data.shape} dtype: {blk_data.dtype}")
    full_screen_dims = (16, 10)
    fig = plt.figure(figsize=full_screen_dims, constrained_layout=True)
    img_save_path = f"./img/spectra/{display_file_name}_blmag.png"
    plot_spectrum_dual_pol(bb_reader,
                           display_file_name=display_file_name,
                           out_filepath=img_save_path,
                           start_freq_mhz=start_freq_mhz,
                           stop_freq_mhz=stop_freq_mhz,
                           center_freq_mhz=center_obs_freq_mhz,
                           flag_show=True
                           )

    # reload the data file for processing, clearing out any reader state
    bb_reader = GuppiRaw(data_file_name)
    n_freq_bins = 128  # TODO calculate a reasonable value
    process_all_channels(bb_reader, n_coarse_chans=n_coarse_chans,
                         n_freq_bins=n_freq_bins,
                         n_data_blocks=n_data_blocks,
                         sampling_rate_hz=sampling_rate_hz,
                         start_freq_mhz=start_freq_mhz,
                         chan_bw_delta_mhz=chan_bw_delta_mhz,
                         display_file_name=display_file_name,
                         )



if __name__ == "__main__":
    main()
