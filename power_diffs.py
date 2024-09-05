"""
Examine filterbank ('.fil') or HDF5 ('.h5') files,
assumed to be from Breakthrough Listen archives
(http://seti.berkeley.edu/opendata)


"""
import sys

import blimpy
import numpy as np
import matplotlib
import argparse

import sigmf
from matplotlib.ticker import StrMethodFormatter
from scipy import ndimage
from sigmf import sigmffile, SigMFFile
from blimpy import Waterfall
import matplotlib.pyplot as plt
import scipy
matplotlib.use('qtagg')


def normalize_power_spectrum(ps):
    ps = (ps - np.mean(ps)) / np.std(ps)
    return ps

def min_max_reduction(matrix, num_segments):
    M, N = matrix.shape
    segment_length = N // num_segments
    print(f"matrix shape: {matrix.shape} num_segs {num_segments} segment_len {segment_length}")

    # Initialize the reduced matrix
    reduced_matrix = np.zeros((M, 2 * num_segments))

    for i in range(M):
        reshaped_row = matrix[i, :segment_length * num_segments].reshape(num_segments, segment_length)

        # Calculate min and max for each segment
        mins = np.min(reshaped_row, axis=1)
        maxs = np.max(reshaped_row, axis=1)

        # Interleave the mins and maxs in the reduced row
        reduced_matrix[i, :] = np.column_stack((mins, maxs)).flatten()

    print(f"reduced matrix shape: {reduced_matrix.shape}")
    return reduced_matrix

def safe_log(data):
    return np.log10(np.abs(data) + sys.float_info.epsilon) * np.sign(data)

def nearest_power2(N):
    # Calculate log2 of N
    a = int(np.log2(N))

    # If 2^a is equal to N, return N
    if 2**a == N:
        return N

    # Return 2^(a + 1)
    return 2**(a + 1)

def main():

    parser = argparse.ArgumentParser(description='Analyze power correlation of filterbank files')
    parser.add_argument('--src_data_path',
                        help="Source hdf5 (.h5) or filerbank (.fil) file path",
                        # default="/Volumes/GLEAN/filterbank/guppi_58410_37136_195860_FRB181017_0001.0000.h5"
                        # default="/Volumes/GLEAN/filterbank/blc20_guppi_57991_66219_DIAG_FRB121102_0020.gpuspec.0001.fil"
                        # default="./data/blc03_samples/blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000.fil"
                        # default="./data/voyager_f1032192_t300_v2.fil"
                        default="./data/voyager1_rosetta_blc3/Voyager1.single_coarse.fine_res.h5"
                        # default="./data/voyager1_rosetta_blc3/Voyager1.single_coarse.fine_res.fil"
                        # default="./data/blc07_samples/blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.fil"
                        )
    args = parser.parse_args()

    data_path = args.src_data_path
    print(f'loading file info: {data_path}')
    obs_obj = blimpy.Waterfall(data_path)
    # dump some interesting facts about this observation
    print(">>> Dump observation info...")
    obs_obj.info()
    # print(f">>> observation header: {obs_obj.header} ")

    if 'rawdatafile' in obs_obj.header:
        print(f"Source raw data file: {obs_obj.header['rawdatafile']}")
    else:
        print(f"No upstream raw data file reported in header")

    # the following reads through the filterbank file in order to calculate the number of coarse channels
    n_coarse_chan = int(obs_obj.calc_n_coarse_chan())
    n_fine_chan = obs_obj.header['nchans']
    fine_channel_bw_mhz = obs_obj.header['foff']

    if fine_channel_bw_mhz < 0: # file data stores descending frequency values
        start_freq = obs_obj.container.f_start - fine_channel_bw_mhz
        stop_freq = obs_obj.container.f_stop
    else: # file data stores ascending frequency values
        start_freq = obs_obj.container.f_start
        stop_freq = obs_obj.container.f_stop - fine_channel_bw_mhz
    print(f"fine_channel_bw_mhz: {fine_channel_bw_mhz} start: {start_freq} stop: {stop_freq}")

    fine_channel_bw_hz = np.abs(fine_channel_bw_mhz) * 1E6
    print(f"n_coarse_chan: {n_coarse_chan} n_fine_chan: {n_fine_chan} fine_channel_bw_hz: {fine_channel_bw_hz:0.3f}")
    n_integrations_input = obs_obj.n_ints_in_file
    n_polarities_stored = obs_obj.header['nifs']

    print(f"data shape: {obs_obj.data.shape} "
          f"s/b: integrations {n_integrations_input}, polarities {n_polarities_stored}, n_fine_chan {n_fine_chan} ")

    # validate that the input file data shape matches expectations set by header
    assert n_integrations_input == obs_obj.data.shape[0]
    assert n_polarities_stored == obs_obj.data.shape[1]
    assert n_fine_chan == obs_obj.data.shape[2]

    # tsamp is "Time integration sampling rate in seconds" (from rawspec)
    integration_period_sec = float(obs_obj.header['tsamp'])
    print(f"Integration period (tsamp): {integration_period_sec} seconds")
    sampling_rate_mhz = np.abs(n_fine_chan * fine_channel_bw_mhz)
    print(f"Sampling bandwidth: {sampling_rate_mhz} MHz")
    sampling_rate_hz = sampling_rate_mhz * 1E6
    # spectra per integration is n
    spectrum_sampling_period = n_fine_chan / sampling_rate_hz
    n_fine_spectra_per_integration = int(np.ceil(integration_period_sec / spectrum_sampling_period))
    # rawspec refers to `Nas` as "Number of fine spectra to accumulate per dump"; defaults to 51, 128, 3072
    print(f"Num fine spectra collected per integration: {n_fine_spectra_per_integration}")
    # n_fine_spectra_per_integration =
    # tsamp           cb_data[i].fb_hdr.tsamp = raw_hdr.tbin * ctx.Nts[i] * ctx.Nas[i]; // Time integration sampling rate in seconds.

    one_sample = obs_obj.data[0,0,0]
    print(f"file nbits: {int(obs_obj.header['nbits'])}")
    print(f"one_sample dtype: {np.dtype(one_sample)} iscomplex: {np.iscomplex(one_sample)}")

    power_diffs = np.zeros((n_integrations_input,n_fine_chan))

    first_ps = obs_obj.data[0,0]
    first_ps_normalized = normalize_power_spectrum(first_ps)
    prior_ps_normalized = first_ps_normalized

    print(f"walking {n_integrations_input} integrations ...")
    for int_idx in range(n_integrations_input):
        cur_ps = obs_obj.data[int_idx,0]
        cur_ps_normalized = normalize_power_spectrum(cur_ps)
        power_diff = np.subtract(cur_ps_normalized, prior_ps_normalized) # x1 - x2
        power_diffs[int_idx] = power_diff
        prior_ps_normalized = cur_ps_normalized

    # we now have a huge array of num_integrations x num_fine_channels,
    # something like 16 x 1048576 difference values

    # downsample_factor = int( (n_fine_chan // n_integrations_input) / 8 )
    # downsample_factor = nearest_power2(downsample_factor)
    downsample_factor =  1024
    print(f"downsample_factor: {downsample_factor} vs n_fine_chan {n_fine_chan} : {downsample_factor}")

    decimated_data = np.array([scipy.signal.decimate(row, downsample_factor, ftype='iir') for row in power_diffs])
    plot_data = safe_log(decimated_data)

    # Plot the decimated data in a couple different cmaps
    fig, axes = plt.subplots(nrows=2, figsize=(12, 8), layout='constrained')
    fig.suptitle(data_path)
    axes[0].imshow(plot_data.T, aspect='auto', cmap='viridis')
    axes[1].imshow(plot_data.T, aspect='auto', cmap='inferno')
    plt.tight_layout()
    plt.show()








if __name__ == "__main__":
    main()
