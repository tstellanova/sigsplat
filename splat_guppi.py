
import argparse

import blimpy
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from blimpy import GuppiRaw
import os
from scipy.signal import welch
from scipy.fftpack import fft

matplotlib.use('qtagg')
MAX_PLT_POINTS = 65536 * 4  # Max number of points in matplotlib plot
# MAX_SIMPLE_POINTS = 65536   # Max number of points in matplotlib plot


def main():
    parser = argparse.ArgumentParser(description='Analyze GUPPI file')
    parser.add_argument('src_path',
                        help="Source raw file path eg 'blc21_guppi.raw' with the '.raw' file extension")
    parser.add_argument('-o', dest='outdir', type=str, default='./data',
                        help='Output directory processed files')

    args = parser.parse_args()
    data_file_name = args.src_path
    out_dir = args.outdir
    print(f"Loading: {data_file_name} , output to: {out_dir}")

    r = GuppiRaw(data_file_name)
    # Seek through the file to find how many data blocks there are in the file
    max_data_blocks = 1
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
    n_chans = int(first_header['OBSNCHAN'])
    chan_bw_mhz = float(first_header['CHAN_BW'])

    # Per davidm, 'CHAN_BW': 2.9296875 (MHz) and 'TBIN': 3.41333333333e-07 should be inverses of each other
    time_bin = float(first_header['TBIN'])
    verify_time_bin = 1 / (np.abs(chan_bw_mhz) * 1E6)
    print(f"verify_tbin: {verify_time_bin} time_bin: {time_bin} ")

    total_obs_bw_mhz = n_chans * np.abs(chan_bw_mhz)
    obs_bw_mhz = np.abs(float(first_header['OBSBW']))
    print(f"n_chans: {n_chans} chan_bw_mhz: {np.abs(chan_bw_mhz)} ")
    print(f"obs_bw_mhz: {obs_bw_mhz} total_obs_bw_mhz: {total_obs_bw_mhz} ")
    assert obs_bw_mhz == total_obs_bw_mhz

    n_pols = int(first_header['NPOL'])
    n_bits = int(first_header['NBITS'])
    block_size = int(first_header['BLOCSIZE'])
    # NTIME = BLOCSIZE * 8 / (2 * NPOL * NCHAN * NBITS)
    ntime_calc = block_size * 8 / ( 2 * n_pols * n_chans * n_bits)
    print(f"NTIME: {ntime_calc} BLOCSIZE: {block_size} NPOL: {n_pols} NCHAN: {n_chans} NBITS: {n_bits}")

    if n_pols > 2:
        n_pols = np.sqrt(n_pols)

    lowest_obs_freq_mhz = highest_obs_freq_mhz - total_obs_bw_mhz
    # desired_channel_idx = 57
    # # r.plot_histogram(flag_show=True)
    # plot_spectrum_hack(r,focus_chan_idx=0, flag_show=True)
    # r.plot_spectrum(flag_show=False)
    r.reset_index()

    focus_freq_min_mhz = 8419.29698
    focus_freq_max_mhz = 8419.32741
    print(f"focus freq range: {focus_freq_min_mhz}... {focus_freq_max_mhz}")
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
    assert first_header['POL_TYPE'] == 'AABBCRCI'
    # For 8 bit samples, each complex sample consists of two bytes:
    # one byte for real followed by one byte for imaginary.

    # voyager shoula appear in here somewhere on  guppi_57650_67573_Voyager1_0002.0000.raw
    desired_channel_idx = 57
    if chan_bw_mhz < 0:
        desired_channel_idx = n_chans - desired_channel_idx

    # desired_low_freq = lowest_obs_freq_mhz + desired_channel_idx * chan_bw_mhz
    # desired_high_freq = desired_low_freq + chan_bw_mhz
    # print(f"desired freq range ({desired_channel_idx}: {desired_low_freq}... {desired_high_freq}")

    print(f"Consuming n_data_blocks: {n_data_blocks}")
    n_blocks_read = 0

    power_normalizer = None

    # After the header, there is a block of binary data containing several samples
    # from each of the output channels of the polyphase filterbank.
    # They are stored as an array ordered by channel, time, and polarization.

    for block_idx in range(0,  n_data_blocks):
        block_hdr, data_pol_A, data_pol_B = r.read_next_data_block_int8()
        if power_normalizer is None:
            n_samples = data_pol_A.shape[1]
            power_normalizer = n_samples * np.abs(chan_bw_mhz) * 1E6

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

        # the data shape returned is (channels, times, complex) ?
        # For 8 bit samples, each complex sample consists of two bytes:
        # one byte for real followed by one byte for imaginary.

        min_pol_A, max_pol_A = np.min(data_pol_A) , np.max(data_pol_A)
        min_pol_B, max_pol_B = np.min(data_pol_B) , np.max(data_pol_B)
        print(f">>> data_pol_A {data_pol_A.shape} {data_pol_A.dtype} min: {min_pol_A} max: {max_pol_A}")
        print(f">>> data_pol_B {data_pol_B.shape} {data_pol_B.dtype} min: {min_pol_B} max: {max_pol_B}")

        focus_chan_pol_A =  data_pol_A[desired_channel_idx]
        focus_chan_pol_B =  data_pol_B[desired_channel_idx]
        print(f"focus_chan_pol_A: {focus_chan_pol_A.shape} min: {np.min(focus_chan_pol_A)} max: {np.max(focus_chan_pol_A)}")
        print(f"focus_chan_pol_B: {focus_chan_pol_B.shape} min: {np.min(focus_chan_pol_B)} max: {np.max(focus_chan_pol_B)}")

        focus_chan_polA_i = focus_chan_pol_A[:,0]
        focus_chan_polA_q = focus_chan_pol_A[:,1]
        print(f"focus_chan_polA_i {focus_chan_polA_i.shape} focus_chan_polA_q {focus_chan_polA_q.shape}")
        # mag_ints = np.sqrt(focus_chan_polA_i**2 + focus_chan_polA_q**2)

        focus_chan_polA_iq = np.dstack( (focus_chan_polA_i, focus_chan_polA_q)).flatten()
        print(f"focus_chan_polA_iq {focus_chan_polA_iq.shape}")
        renorm_iq = focus_chan_polA_iq.astype(np.float32)
        iq_view = renorm_iq.view('complex64')
        # iq_view_mag = np.abs(iq_view)

        print(f"renorm_iq : {renorm_iq.shape}")
        print(f"iq_view {iq_view.shape}")

        N = len(iq_view)
        # use a larger output bucket to force padding with zeroes (fixes circularity)
        block_fft = np.fft.fft(iq_view, n=N * 2)
        power_spectrum = block_fft * np.conj(block_fft)
        power_spectrum[:5] = 0
        power_spectrum[-5:] = 0

        n_blocks_read += 1

        subplot_rows = 2
        subplot_row_idx = 0
        fig, axs = plt.subplots(subplot_rows, 1,  figsize=(16, 8))

        plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
        plt.plot(power_spectrum)
        # plt.axvline(chan_ctr_freq_mhz, color="skyblue")
        # plt.axvline(8420.21645, color="red")
        # plt.xlabel('Frequency (MHz)')
        plt.ylabel('PSD (norm)')
        plt.grid(True)

        # plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
        # # plt.axvline(chan_ctr_freq_mhz, color="skyblue")
        # plt.plot(iq_view_mag)
        # # plt.axvline(8420.21645, color="red")
        # # plt.xlabel('Frequency (MHz)')
        # plt.ylabel('Mag (complex)')
        # plt.grid(True)

        plt.show()


    # subplot_rows = 2
    # subplot_row_idx = 0
    # # plt.figure(figsize=(16, 8))
    # fig, axs = plt.subplots(subplot_rows, 1,  figsize=(16, 8))
    # plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
    # plt.axvline(chan_ctr_freq_mhz, color="skyblue")
    # plt.plot(freqs0 , PSD0_avg)
    # plt.axvline(8420.21645, color="red")
    # plt.xlabel('Frequency (MHz)')
    # plt.ylabel('Power/Frequency (dB/Hz)')
    # plt.grid(True)
    # plt.subplot(subplot_rows, 1, (subplot_row_idx:=subplot_row_idx+1))
    # plt.axvline(chan_ctr_freq_mhz, color="skyblue")
    # plt.plot(freqs1 , PSD1_avg)
    # plt.axvline(8420.21645, color="red")
    # plt.xlabel('Frequency (MHz)')
    # plt.ylabel('Power/Frequency (dB/Hz)')
    # plt.grid(True)
    #
    # plt.show()

    # base_path = os.path.splitext(os.path.basename(data_file_name))[0]
    # img_base_path = os.path.join(img_out_dir, base_path)
    # plt.savefig()
    # r.plot_spectrum(filename="%s_spec.png" % img_base_path, plot_db=True, flag_show=True)


if __name__ == "__main__":
    main()
