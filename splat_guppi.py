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
import scipy
from blimpy import GuppiRaw
import os
from scipy.signal import welch
from scipy.fftpack import fft
from scipy.ndimage import gaussian_filter1d

matplotlib.use('qtagg')
MAX_PLT_POINTS = 65536 * 4  # Max number of points in matplotlib plot

def safe_log(data):
    return np.log10(np.abs(data) + sys.float_info.epsilon) * np.sign(data)
def gaussian_decimate(data, num_out_buckets, sigma=1.0):
    # Step 1: Apply Gaussian filter
    filtered_data = gaussian_filter1d(data, sigma=sigma)

    # Step 2: Decimate (downsample)
    decimation_factor = len(data) // num_out_buckets
    decimated_data = filtered_data[::decimation_factor]

    return decimated_data

def process_dual_pol_block(n_chans=0, data_pol_a=None, data_pol_b=None):

    min_pol_a, max_pol_a = np.min(data_pol_a) , np.max(data_pol_a)
    min_pol_b, max_pol_b = np.min(data_pol_b) , np.max(data_pol_b)
    # print(f">>> data_pol_a {data_pol_a.shape} {data_pol_a.dtype} min: {min_pol_a} max: {max_pol_a}")
    # print(f">>> data_pol_B {data_pol_b.shape} {data_pol_b.dtype} min: {min_pol_b} max: {max_pol_b}")

    samples_per_block = data_pol_a.shape[1]
    output_shape = (n_chans, samples_per_block)
    ffts = np.zeros(output_shape, dtype=np.complex64)
    psds = np.zeros(output_shape, dtype=np.float32)
    # window_func = np.hamming(samples_per_block)
    for chan_idx in range(n_chans):
        chan_pol_a =  data_pol_a[chan_idx]
        chan_pol_b =  data_pol_b[chan_idx]
        # print(f"chan_pol_a: {chan_pol_a.shape} min: {np.min(chan_pol_a)} max: {np.max(chan_pol_a)}")
        # print(f"chan_pol_b: {chan_pol_b.shape} min: {np.min(chan_pol_b)} max: {np.max(chan_pol_b)}")

        # For 8 bit samples, each complex sample consists of two bytes:
        # one byte for real followed by one byte for imaginary.
        chan_pol_a_iq = chan_pol_a.flatten()
        chan_pol_b_iq = chan_pol_b.flatten()
        # print(f"chan_pol_a_iq {chan_pol_a_iq.shape}")

        # renorm_iq = chan_pol_a_iq.astype(np.float32)
        # print(f"renorm_iq: {renorm_iq.shape}")

        # TODO can we jump straight to view complex64 without recasting?
        chan_pol_a_view = chan_pol_a_iq.astype(np.float32).view('complex64')
        # print(f"chan_pol_a_view {chan_pol_a_view.shape}")
        chan_pol_b_view = chan_pol_b_iq.astype(np.float32).view('complex64')

        N = len(chan_pol_a_view)
        assert N == samples_per_block
        eff_N = scipy.fft.next_fast_len(N)

        # Apply window to taper edges to zero
        # TODO alternately use a larger FFT?
        # chan_pol_a_view = chan_pol_a_view * window_func
        # chan_pol_b_view = chan_pol_b_view * window_func

        # use a larger output bucket to force padding with zeroes (fixes circularity)
        # chan_pol_a_fft = fft(chan_pol_a_view, n=N * 2)
        # chan_pol_a_fft = fft(chan_pol_a_view)
        # chan_pol_b_fft = fft(chan_pol_b_view)
        chan_pol_a_fft = np.fft.fft(chan_pol_a_view, n=eff_N) #n=N * 2)
        fft_res_len = len(chan_pol_a_fft)
        # print(f"original N: {N} / fft_res_len: {fft_res_len}")
        chan_pol_a_fft = chan_pol_a_fft[:N]
        chan_pol_b_fft = np.fft.fft(chan_pol_b_view, n=eff_N) #n=N * 2)
        chan_pol_b_fft = chan_pol_b_fft[:N]

        chan_pol_a_psd = chan_pol_a_fft * np.conj(chan_pol_a_fft)
        # print(f"chan_pol_a_fft: {chan_pol_a_fft.shape}  chan_pol_a_psd: {chan_pol_a_psd.shape}")
        chan_pol_b_psd = chan_pol_b_fft * np.conj(chan_pol_b_fft)
        # add power from both polarities to get total power
        sum_psd = np.abs(chan_pol_b_psd + chan_pol_a_psd)
        # TODO useful? ffts[chan_idx] = chan_pol_a_fft
        psds[chan_idx] = sum_psd

    min_psd = np.min(psds)
    max_psd = np.max(psds)
    print(f"min_psd: {min_psd:0.3e} max_psd: {max_psd:0.3e}")

    return ffts, psds


def main():
    parser = argparse.ArgumentParser(description='Analyze GUPPI file')
    parser.add_argument('src_path', nargs='?',
                        help="Source raw file path eg 'blc21_guppi.raw' with the '.raw' file extension",
                        default="../../baseband/bloda/"
                                "blc4_2bit_guppi_57432_24865_PSR_J1136+1551_0002.0001.raw"
                        )
    parser.add_argument('-o', dest='outdir', type=str, default='./data',
                        help='Output directory processed files')

    args = parser.parse_args()
    data_file_name = args.src_path
    out_dir = args.outdir
    print(f"Loading: {data_file_name} , output to: {out_dir}")

    r = GuppiRaw(data_file_name)
    # Seek through the file to find how many data blocks there are in the file
    max_data_blocks = 256
    n_data_blocks = r.find_n_data_blocks()
    print(f"total n_data_blocks: {n_data_blocks}")
    if n_data_blocks > max_data_blocks:
        print(f"clamping to {max_data_blocks} blocks")
        n_data_blocks = max_data_blocks

    # reads the first header and then seek to file offset zero
    first_header = r.read_first_header()
    print(f"first_header: {first_header}")
    highest_obs_freq_mhz = float(first_header['OBSFREQ'])
    print(f"highest_obs_freq_mhz {highest_obs_freq_mhz} MHz")
    n_coarse_chans = int(first_header['OBSNCHAN'])
    chan_bw_mhz = float(first_header['CHAN_BW'])

    # Per davidm, 'CHAN_BW': 2.9296875 (MHz) and 'TBIN': 3.41333333333e-07 should be inverses of each other
    time_bin = float(first_header['TBIN'])
    verify_time_bin = 1 / (np.abs(chan_bw_mhz) * 1E6)
    print(f"verify_tbin - time_bin: {verify_time_bin - time_bin:0.3e}")
    # assert verify_time_bin == time_bin

    total_obs_bw_mhz = n_coarse_chans * np.abs(chan_bw_mhz)
    obs_bw_mhz = np.abs(float(first_header['OBSBW']))
    print(f"n_coarse_chans: {n_coarse_chans} chan_bw_mhz: {chan_bw_mhz} total_obs_bw_mhz: {total_obs_bw_mhz}")
    # print(f"obs_bw_mhz: {obs_bw_mhz} total_obs_bw_mhz: {total_obs_bw_mhz} ")
    assert obs_bw_mhz == total_obs_bw_mhz

    n_pols = int(first_header['NPOL'])
    n_bits = int(first_header['NBITS'])
    block_size = int(first_header['BLOCSIZE'])
    # NTIME = BLOCSIZE * 8 / (2 * NPOL * NCHAN * NBITS)
    ntime_calc = block_size * 8 / ( 2 * n_pols * n_coarse_chans * n_bits)
    print(f"NTIME: {ntime_calc} BLOCSIZE: {block_size} NPOL: {n_pols} NCHAN: {n_coarse_chans} NBITS: {n_bits}")

    if n_pols > 2:
        n_pols = np.sqrt(n_pols)

    lowest_obs_freq_mhz = highest_obs_freq_mhz - total_obs_bw_mhz
    print(f"avail freq range: {lowest_obs_freq_mhz}... {highest_obs_freq_mhz}")

    r.reset_index()

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
    assert first_header['POL_TYPE'] == 'AABBCRCI'
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

    print(f"Consuming n_data_blocks: {n_data_blocks} n_coarse_chans: {n_coarse_chans}")
    n_blocks_read = 0

    focus_coarse_chan_idx = int(n_coarse_chans // 2)
    focus_chan_psd_diffs = None
    prior_psd = None
    for block_idx in range(0,  n_data_blocks):
        block_hdr, data_pol_A, data_pol_B = r.read_next_data_block_int8()
        _ffts, psds = process_dual_pol_block(n_coarse_chans, data_pol_A, data_pol_B)
        n_blocks_read += 1
        if focus_chan_psd_diffs is None:
            focus_chan_psd_diffs = np.zeros((n_data_blocks, psds.shape[1]))
            print(f"focus_chan_psd_diffs: {focus_chan_psd_diffs.shape}")



        # only save the psd from the focus coarse channel
        cur_psd = psds[focus_coarse_chan_idx]
        if prior_psd is None:
            prior_psd = cur_psd
            continue
        cur_psd_diff = np.subtract(cur_psd, prior_psd)
        focus_chan_psd_diffs[block_idx] = cur_psd_diff
        # print(f"psds: {focus_chan_psd_diffs[block_idx].shape}")
        prior_psd = cur_psd

    down_buckets =  1024
    dec_diffs = np.array([gaussian_decimate(row, down_buckets) for row in focus_chan_psd_diffs])
    psd_diffs_scaled = 10*safe_log(dec_diffs)

    psd_diffs_scaled[np.abs(psd_diffs_scaled) < 50] = 0
    # thresholded_psd_diffs = np.where( np.abs(psd_diffs_scaled) < 50, 0, psd_diffs_scaled)
    # thresholded_psd_diffs = np.clip(thresholded_psd_diffs, a_min=-50, a_max=50)

    while True:
        full_screen_dims=(16, 10)
        fig, axes = plt.subplots(nrows=2, figsize=full_screen_dims,  constrained_layout=True,  sharex=True)
        # fig.subplots_adjust(hspace=0)
        fig.suptitle(f"{lowest_obs_freq_mhz:0.4f} | {data_file_name}")

        img0 = axes[0].imshow(psd_diffs_scaled, aspect='auto', cmap='viridis')
        axes[0].set_ylabel('Block')
        img1 = axes[1].imshow(psd_diffs_scaled, aspect='auto', cmap='inferno')
        axes[1].set_ylabel('Block')
        axes[1].set_xlabel('Freq Bin')
        cbar0 = fig.colorbar(img0, ax=axes[0])
        cbar0.set_label('ΔPSD dB', rotation=270)

        cbar1 = fig.colorbar(img1, ax=axes[1])
        cbar1.set_label('ΔPSD dB', rotation=270)

        plt.show()
        return




if __name__ == "__main__":
    main()
