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
import os
import logging

import matplotlib.pyplot as plt

from sigsplat.convert import fbank, spectralize

matplotlib.use('qtagg')

# performance monitoring
from time import perf_counter


def main():

    parser = argparse.ArgumentParser(description='Analyze power correlation of filterbank files')
    parser.add_argument('src_data_path', nargs='?',
                        help="Source hdf5 (.h5) or filerbank (.fil) file path",
                        default="../../filterbank/blgcsurvey_cband/spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58705_13603_BLGCsurvey_Cband_C12_0058.gpuspec.0002.fil"
                        # default="../../filterbank/blc27_guppi_58410_37136_195860_FRB181017_0001.0000.h5"
                        # default="../../filterbank/blc20_guppi_57991_66219_DIAG_FRB121102_0020.gpuspec.0001.fil"
                        # default="../../filterbank/guppi_58410_37136_195860_FRB181017_0001.0000.h5"
                        # default="../../filterbank/blc20_guppi_57991_66219_DIAG_FRB121102_0020.gpuspec.0001.fil"
                        # default="./data/blc03_samples/blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000.fil"
                        # default="./data/voyager_f1032192_t300_v2.fil" k
                        # default="../../filterbank/voyager1_rosetta_blc3/Voyager1.single_coarse.fine_res.h5" k
                        # default="../../filterbank/voyager1_rosetta_blc3/Voyager1.single_coarse.fine_res.fil"
                        # default="./data/blc07_samples/blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.fil" k
                        )
    args = parser.parse_args()

    data_path = args.src_data_path
    print(f'Loading file info: {data_path} ...')
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
    loggers = [logging.getLogger()]  # get the root logger
    loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    print(f"Loggers: {loggers}")
    # logging.getLogger("root").setLevel(logging.DEBUG)
    logging.getLogger("blimpy.io.base_reader").setLevel(logging.DEBUG)
    logging.getLogger("blimpy").setLevel(logging.DEBUG)

    perf_start = perf_counter()
    obs_obj = blimpy.Waterfall(data_path, max_load=16, load_data=False)

    print(f"elapsed: {perf_counter()  - perf_start:0.3f} seconds")

    # dump some interesting facts about this observation
    print(">>> Dump observation info ...")
    perf_start = perf_counter()
    obs_obj.info()
    print(f"elapsed: {perf_counter()  - perf_start:0.3f} seconds")

    if 'rawdatafile' in obs_obj.header:
        print(f"Source raw data file: {obs_obj.header['rawdatafile']}")
    else:
        print(f"No upstream raw data file reported in header")

    src_data_filename = os.path.basename(data_path)

    # the following reads through the filterbank file in order to calculate the number of coarse channels
    print("Calculate coarse channels ...")
    perf_start = perf_counter()
    n_coarse_chan = int(obs_obj.calc_n_coarse_chan())
    print(f"elapsed: {perf_counter()  - perf_start:0.3f} seconds")

    n_fine_chan = obs_obj.header['nchans']
    fine_channel_bw_mhz = obs_obj.header['foff']

    if fine_channel_bw_mhz < 0: # file data stores descending frequency values
        start_freq = obs_obj.container.f_start - fine_channel_bw_mhz
        stop_freq = obs_obj.container.f_stop
    else: # file data stores ascending frequency values
        start_freq = obs_obj.container.f_start
        stop_freq = obs_obj.container.f_stop - fine_channel_bw_mhz
    print(f"fine_channel_bw_mhz: {fine_channel_bw_mhz:0.6e} start: {start_freq:0.6f} stop: {stop_freq:0.6f}")

    fine_channel_bw_hz = np.abs(fine_channel_bw_mhz) * 1E6
    print(f"n_coarse_chan: {n_coarse_chan} n_fine_chan: {n_fine_chan} fine_channel_bw_hz: {fine_channel_bw_hz:0.4f}")
    n_integrations_input = obs_obj.n_ints_in_file
    n_polarities_stored = obs_obj.header['nifs']
    # We don't actually handle anything other than Stokes I at the moment
    assert n_polarities_stored == 1

    # if obs_obj.container.isheavy():
    #     selection_size_bytes =  obs_obj.container._calc_selection_size()
    #     selection_shape = obs_obj.container.selection_shape
    #     data_array_size = obs_obj.container.max_data_array_size
    #     print(f"heavy selection_shape: {selection_shape}  selection_size_bytes: {selection_size_bytes} data_array_size: {data_array_size}")

    # print(f"data shape: {obs_obj.data.shape} "
    #       f"s/b: integrations {n_integrations_input}, polarities {n_polarities_stored}, n_fine_chan {n_fine_chan} ")

    # validate that the input file data shape matches expectations set by header
    # # TODO this fails for large files
    # assert n_integrations_input == obs_obj.data.shape[0]
    # assert n_polarities_stored == obs_obj.data.shape[1]
    # assert n_fine_chan == obs_obj.data.shape[2]

    n_integrations_to_process = n_integrations_input
    if n_integrations_input > 256:
        n_integrations_to_process = 256
        print(f"clamping n_integrations_to_process to {n_integrations_to_process}")

    # tsamp is "Time integration sampling rate in seconds" (from rawspec)
    integration_period_sec = float(obs_obj.header['tsamp'])
    print(f"Integration period (tsamp): {integration_period_sec:0.6f} seconds")
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

    print(f"file nbits: {int(obs_obj.header['nbits'])}")

    power_diffs = np.zeros((n_integrations_to_process,n_fine_chan))

    print(f"Load first integration from disk...")
    perf_start = perf_counter()
    obs_obj.read_data(t_start=0, t_stop=1)
    _, first_ps = obs_obj.grab_data(t_start=0, t_stop=1)
    print(f"elapsed: {perf_counter()  - perf_start:0.3f} seconds")
    print(f"one integration shape: {first_ps.shape} ")
    prior_ps_normalized = spectralize.normalize_data(first_ps)

    print(f"Walking {n_integrations_to_process} integrations ...")
    perf_start = perf_counter()
    for int_idx in range(n_integrations_to_process):
        print(f"integration: {int_idx}")
        cur_ps = fbank.grab_one_integration(obs_obj, int_idx)
        cur_ps_normalized =  spectralize.normalize_data(cur_ps)
        cur_diff = np.subtract(cur_ps_normalized, prior_ps_normalized) # x1 - x2
        power_diffs[int_idx] = np.sign(cur_diff) * (cur_diff * cur_diff)
        prior_ps_normalized = cur_ps_normalized
        cur_ps = None

    prior_ps_normalized = None
    print(f"elapsed: {perf_counter()  - perf_start:0.3f} seconds")

    # we now have a huge array of num_integrations x num_fine_channels,
    # something like 16 x 1048576 difference values
    # need to decimate to make platting work

    down_buckets = 1024

    if n_fine_chan > down_buckets:
        print(f"Decimating {n_fine_chan} fine channels to {down_buckets} buckets, ratio: {n_fine_chan / down_buckets} ...")
        perf_start = perf_counter()
        # decimated_data = np.array([scipy.signal.decimate(row, down_buckets, ftype='iir') for row in power_diffs])
        # decimated_data = np.array([scipy.signal.decimate(row, down_buckets, ftype='fir') for row in power_diffs])
        decimated_data = np.array([spectralize.gaussian_decimate(row, num_out_buckets=down_buckets) for row in power_diffs])
        print(f"elapsed: {perf_counter()  - perf_start:0.3f} seconds")
    else:
        down_buckets = n_fine_chan
        decimated_data = power_diffs

    power_diffs = None
    # plot_data = decimated_data
    plot_data = spectralize.safe_scale_log(decimated_data)
    decimated_data = None

    print(f"Plotting {plot_data.shape} ...")
    perf_start = perf_counter()

    # Plot the decimated data in a couple different cmaps
    # my_dpi=300
    # full_screen_dims=(3024 / my_dpi, 1964 / my_dpi)
    full_screen_dims=(16, 10)
    fig, axes = plt.subplots(nrows=2, figsize=full_screen_dims,  constrained_layout=True,  sharex=True)
    fig.subplots_adjust(hspace=0)
    fig.suptitle(f"{start_freq:0.4f} | {src_data_filename}")

    img0 = axes[0].imshow(plot_data.T, aspect='auto', cmap='viridis')
    img1 = axes[1].imshow(plot_data.T, aspect='auto', cmap='inferno')

    cbar0 = fig.colorbar(img0, ax=axes[0])
    cbar0.set_label('Δ dB', rotation=270)

    cbar1 = fig.colorbar(img1, ax=axes[1])
    cbar1.set_label('Δ dB', rotation=270)

    yticks = axes[0].get_yticks()
    # print(f"axes0 yticks {yticks}")
    tick_step_mhz = sampling_rate_mhz / len(yticks)
    reduced_freqs = np.arange(start_freq, stop_freq, tick_step_mhz)
    reduced_freqs -= start_freq
    ytick_labels = [f"{freq:0.4f}" for freq in reduced_freqs]
    # print(f"ytick_labels: {ytick_labels}")
    axes[0].set_yticklabels(ytick_labels)
    axes[1].set_yticklabels(ytick_labels)
    axes[0].set_ylabel('BB MHz')
    axes[1].set_ylabel('BB MHz')

    print(f"elapsed: {perf_counter()  - perf_start:0.3f} seconds")
    img_save_path = f"./plots/{src_data_filename}.powdiff.png"
    print(f"saving image to:\n{img_save_path}")
    plt.savefig(img_save_path)

    plt.show()


if __name__ == "__main__":
    main()
