"""
Examine filterbank ('.fil') or HDF5 ('.h5') files,
assumed to be from Breakthrough Listen archives
(http://seti.berkeley.edu/opendata)


"""

import blimpy
import numpy as np
import matplotlib
import argparse

import sigmf
from matplotlib.ticker import StrMethodFormatter
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


def main():

    parser = argparse.ArgumentParser(description='Analyze power correlation of filterbank files')
    parser.add_argument('--src_data_path',
                        help="Source hdf5 (.h5) or filerbank (.fil) file path",
                        default="/Volumes/GLEAN/filterbank/guppi_58410_37136_195860_FRB181017_0001.0000.h5"
                        # default="/Volumes/GLEAN/filterbank/blc20_guppi_57991_66219_DIAG_FRB121102_0020.gpuspec.0001.fil"
                        # default="./data/blc03_samples/blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000.fil"
                        # default="./data/voyager_f1032192_t300_v2.fil"
                        # default="./data/voyager1_rosetta_blc3/Voyager1.single_coarse.fine_res.h5"
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
    # two_sample = obs_obj.data[0,0,0]
    # print(f"two_sample: {two_sample} dtype: {np.dtype(two_sample)} iscomplex: {np.iscomplex(two_sample)}")

    delta_correlations = np.zeros((n_integrations_input, n_fine_chan))
    first_correlations = np.zeros((n_integrations_input, n_fine_chan))

    first_ps = obs_obj.data[0,0]
    first_ps_normalized = normalize_power_spectrum(first_ps)
    prior_ps_normalized = first_ps_normalized
    # print(f"first shapes: {first_ps.shape} {first_ps_normalized.shape}")
    print(f"walking {n_integrations_input} integrations")
    for int_idx in range(n_integrations_input):
        cur_ps = obs_obj.data[int_idx,0]
        cur_ps_normalized = normalize_power_spectrum(cur_ps)
        first_correlations[int_idx] = scipy.signal.correlate(first_ps_normalized, cur_ps_normalized, mode='valid')
        delta_correlations[int_idx] = scipy.signal.correlate(prior_ps_normalized, cur_ps_normalized, mode='valid')
        prior_ps_normalized = cur_ps_normalized

    print(f"Massaging correlations...")
    first_ultamax = np.max(np.max(first_correlations,axis=1))
    first_correlations = np.divide(first_correlations, first_ultamax)
    # delta_ultamax = np.max(np.max(delta_correlations,axis=1))
    delta_correlations = np.divide(delta_correlations, first_ultamax)

    med_first_corrs = np.median(first_correlations,axis=1)
    med_delta_corrs = np.median(delta_correlations,axis=1)

    fig, axes = plt.subplots(nrows=2, figsize=(12, 8), layout='constrained')
    fig.suptitle(data_path)
    axes[0].plot(med_first_corrs, label="firsts")
    axes[0].legend()
    axes[1].plot(med_delta_corrs[1:], label="deltas")
    axes[1].legend()
    plt.show()
    return

    n_plot_freq_bins = int(32)
    # plot_freqs = np.linspace(start_freq, stop_freq, n_plot_freq_bins)
    decimation_factor = int(np.round(n_fine_chan / n_plot_freq_bins))
    print(f"decimation_factor: {decimation_factor}")


    # first_corrs_dec = scipy.signal.decimate(first_correlations, decimation_factor)
    # delta_corrs_dec = scipy.signal.decimate(delta_correlations, decimation_factor)

    first_corrs_dec = min_max_reduction(first_correlations, n_plot_freq_bins)
    delta_corrs_dec = min_max_reduction(delta_correlations, n_plot_freq_bins)

    # print(f"first_corrs_dec: {first_corrs_dec.shape} {first_corrs_dec[1]}")
    # print(f"delta_corrs_dec: {delta_corrs_dec.shape} {delta_corrs_dec[1]}")

    # overall_corr_min = np.min([min_first_corr, min_delta_corr])
    # overall_corr_max = np.max([max_first_corr, max_delta_corr])

    print(f"first corr dec: {np.min(first_corrs_dec)} .. {np.max(first_corrs_dec)}")
    print(f"delta corr dec: {np.min(delta_corrs_dec)} .. {np.max(delta_corrs_dec)}")

    # plot the parallel correlations
    fig, axes = plt.subplots(nrows=2, figsize=(12, 8), layout='constrained')
    cmap = plt.colormaps["plasma"]
    cmap = cmap.with_extremes(bad=cmap(0))

    pcm0 = axes[0].pcolormesh(first_corrs_dec.T,
                             norm="log",
                             # norm="linear",
                             # vmax=overall_max,
                             # vmin=min_first_corr, vmax=max_first_corr,
                             cmap=cmap, rasterized=True)
    fig.colorbar(pcm0, ax=axes[0])

    # pcm1 = axes[1].pcolormesh(delta_corrs_dec.T,
    #                          norm="log",
    #                          # norm="linear",
    #                          # vmax=overall_max,
    #                          # vmin=min_delta_corr, vmax=max_delta_corr,
    #                          cmap=cmap, rasterized=True)
    # fig.colorbar(pcm1, ax=axes[1])

    plt.show()


if __name__ == "__main__":
    main()
