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

import sigmf
from matplotlib.ticker import StrMethodFormatter
from scipy import ndimage
from scipy.signal import find_peaks
from sigmf import sigmffile, SigMFFile
from blimpy import Waterfall
import matplotlib.pyplot as plt
import scipy
matplotlib.use('qtagg')

# performance monitoring
from time import perf_counter


from scipy.ndimage import gaussian_filter1d

def round_to_nearest_power_of_two(n):
    return int(np.power(2, np.round(np.log2(n))))

def normalize_data(data):
    return (data - np.mean(data)) / np.std(data)

def remove_peaks_vectorized(arr_db, db_threshold=5):

    # Identify peaks: Values greater than both their left and right neighbors
    left_shift = np.roll(arr_db, 1)
    right_shift = np.roll(arr_db, -1)

    is_peak = (arr_db > left_shift) & (arr_db > right_shift)

    # Calculate the average of the neighboring values for each sample
    surrounding_avg = (left_shift + right_shift) / 2

    # Find peaks that exceed the threshold compared to the neighboring average
    exceeds_threshold = (arr_db - surrounding_avg) >= db_threshold

    # Get the positions of peaks that meet both conditions
    peaks_to_remove = is_peak & exceeds_threshold

    # Replace the peaks with the surrounding average
    arr_db[peaks_to_remove] = surrounding_avg[peaks_to_remove]

    return arr_db

def safe_log(data):
    return np.log10(np.abs(data) + sys.float_info.epsilon) * np.sign(data)

def simple_decimate(data, num_out_buckets):
    decimation_factor = len(data) // num_out_buckets
    decimated_data = data[::decimation_factor]
    return decimated_data

def gaussian_decimate(data, num_out_buckets, sigma=1.0):
    filtered_data = gaussian_filter1d(data, sigma=sigma)
    return simple_decimate(filtered_data, num_out_buckets)

def grab_one_integration(obs_obj:Waterfall=None, integration_idx=0):
    obs_obj.read_data(t_start=integration_idx, t_stop=integration_idx+1)
    _, cur_ps = obs_obj.grab_data()
    return cur_ps

def main():

    parser = argparse.ArgumentParser(description='Analyze power correlation of filterbank files')
    parser.add_argument('src_data_path', nargs='?',
                        help="Source hdf5 (.h5) or filerbank (.fil) file path",
                        # default="../../filterbank/blgcsurvey_cband/"
                            # "spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58705_13603_BLGCsurvey_Cband_C12_0058.gpuspec.0000.fil" # LORGE 16 ints, 1664 coarse, 1744830464 fine
                            # "spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58705_13603_BLGCsurvey_Cband_C12_0058.gpuspec.0002.fil" # 279 ints, 1664 coarse,
                            # "spliced_blc00010203040506o7o01113141516o7o0212223242526o7o031323334353637_guppi_58705_14221_BLGCsurvey_Cband_C12_0060.gpuspec.0002.fil" # 279 ints, 1664 coarse, 1703936 fine
                            #  "spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58705_18741_BLGCsurvey_Cband_A00_0063.gpuspec.0002.fil" # 120 ints, 1164 coarse, 1703936 fine
                            # "spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58705_01220_BLGCsurvey_Cband_B04_0018.gpuspec.0002.fil"
                            # "spliced_blc40414243444546o7o0515253545556o7o0616263646566o7o071727374757677_guppi_58702_18718_BLGCsurvey_Cband_C01_0026.gpuspec.0002.fil"

                        default="../../filterbank/misc/"
                                # "blc20_guppi_57991_66219_DIAG_FRB121102_0020.gpuspec.0001.fil" #  64 coarse chan, 512 fine, 3594240 integrations?
                                # "voyager_f1032192_t300_v2.fil" # 2 integrations, 63 coarse channels, small
                          "blc27_guppi_58410_37136_195860_FRB181017_0001.0000.h5" # 44 coarse chans, 78 integrations
                        # default="../../filterbank/blc07/"
                        #   "blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.fil" # 16 integrations
                        # default="../../filterbank/blc03/"
                                # "blc3_2bit_guppi_57386_VOYAGER1_0002.gpuspec.0000.fil" # 16 integrations, 64 coarse, 66125824 fine
                        # default="../../filterbank/voyager1_rosetta_blc3/"
                        #    "Voyager1.single_coarse.fine_res.h5" # 16 integrations, single coarse channel
                        #   "Voyager1.single_coarse.fine_res.fil"
                        )
    parser.add_argument('-o', dest='plots_path', type=str, default='./plots/',
                        help='Output directory for plots')

    args = parser.parse_args()

    # freq_of_interest = 8419.2972
    data_path = args.src_data_path
    plots_path = args.plots_path

    print(f'Loading file info: {data_path} ...')
    logging.basicConfig(level=os.environ.get("LOGLEVEL", "DEBUG"))
    loggers = [logging.getLogger()]  # get the root logger
    loggers = loggers + [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    # print(f"Loggers: {loggers}")
    # logging.getLogger("root").setLevel(logging.DEBUG)
    logging.getLogger("blimpy.io.base_reader").setLevel(logging.DEBUG)
    logging.getLogger("blimpy").setLevel(logging.DEBUG)

    full_screen_dims=(16, 10)

    perf_start = perf_counter()
    obs_obj = blimpy.Waterfall(data_path, max_load=16) #, load_data=False)
    # focus_freq_start = 8420.2163
    # focus_freq_stop = 8420.2166
    # obs_obj.read_data(f_start=focus_freq_start, f_stop=focus_freq_stop) # Voyager carrier only

    # plt.figure(figsize=full_screen_dims)
    # obs_obj.plot_spectrum(logged=True)
    # plt.show()

    # plt.figure(figsize=full_screen_dims)
    # obs_obj.plot_waterfall()
    # plt.show()

    # dump some interesting facts about this observation
    obs_obj.info()

    src_data_filename = os.path.basename(data_path)
    display_file_name = os.path.splitext(os.path.basename(src_data_filename))[0]

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

    coarse_channel_bw_mhz = np.abs(stop_freq - start_freq)/n_coarse_chan
    print(f"coarse_channel_bw_mhz: {coarse_channel_bw_mhz}")
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
    # # TODO these checks fail for large files becaue data is not preloaded
    # assert n_integrations_input == obs_obj.data.shape[0]
    # assert n_polarities_stored == obs_obj.data.shape[1]
    # assert n_fine_chan == obs_obj.data.shape[2]

    n_integrations_to_process = n_integrations_input
    if n_integrations_input > 280:
        n_integrations_to_process = 280
        print(f"clamping n_integrations_to_process to {n_integrations_to_process}")

    # tsamp is "Time integration sampling rate in seconds" (from rawspec)
    integration_period_sec = float(obs_obj.header['tsamp'])
    print(f"Integration period (tsamp): {integration_period_sec:0.6f} seconds")
    total_sampling_rate_mhz = np.abs(n_fine_chan * fine_channel_bw_mhz)
    print(f"Total sampling bandwidth: {total_sampling_rate_mhz} MHz")
    sampling_rate_hz = total_sampling_rate_mhz * 1E6
    # spectra per integration is n
    spectrum_sampling_period = n_fine_chan / sampling_rate_hz
    n_fine_spectra_per_integration = int(np.ceil(integration_period_sec / spectrum_sampling_period))
    # rawspec refers to `Nas` as "Number of fine spectra to accumulate per dump"; defaults to 51, 128, 3072
    print(f"Num fine spectra accumulated per integration: {n_fine_spectra_per_integration}")
    # n_fine_spectra_per_integration =
    # tsamp   cb_data[i].fb_hdr.tsamp = raw_hdr.tbin * ctx.Nts[i] * ctx.Nas[i]; // Time integration sampling rate in seconds.

    file_nbits = int(obs_obj.header['nbits'])
    # print(f"file nbits: {file_nbits}")
    assert file_nbits == 32

    print(f"Load first integration from disk...")
    perf_start_first_int_load = perf_counter()
    print(f"shape of one integration: {obs_obj.data.shape}")
    obs_freqs, prior_ps = obs_obj.grab_data(t_start=0, t_stop=1)
    # prior_ps /= n_fine_spectra_per_integration # average value of the PS over the integration
    print(f"obs_freqs {obs_freqs[0]} ... {obs_freqs[-1]}")
    print(f"First integration load >>> elapsed: {perf_counter()  - perf_start_first_int_load:0.3f} seconds")
    print(f"first integration shape: {prior_ps.shape} {prior_ps.dtype} min: {np.min(prior_ps):0.3e} max: {np.max(prior_ps):0.3e}")

    n_input_buckets = len(prior_ps)
    if n_input_buckets != n_fine_chan:
        print(f"processing subset: {n_input_buckets} / {n_fine_chan} fine channels")

    raw_psds = np.zeros((n_integrations_to_process, n_input_buckets), dtype=np.float32 )

    print(f"Collecting {n_integrations_to_process} integrations ...")
    perf_start_collection = perf_counter()
    for int_idx in range(n_integrations_to_process):
        cur_ps = grab_one_integration(obs_obj, int_idx)
        # convert power counts to db
        cur_ps = safe_log(cur_ps)
        # The filt data often has one or more narrow "DC-offset" peaks -- we attempt to remove those here
        cur_ps = remove_peaks_vectorized(cur_ps, db_threshold=3)
        # this performs a low-pass filter on changes between adjacent frequency bins
        cur_ps = gaussian_filter1d(cur_ps, sigma=3)
        raw_psds[int_idx] = cur_ps
    print(f"Collection >>> elapsed: {perf_counter() - perf_start_collection:0.3f} seconds")

    min_raw_psd = np.min(raw_psds)
    max_raw_psd = np.max(raw_psds)

    print(f"Raw PSDs range: {min_raw_psd:0.3e} ... {max_raw_psd:0.3e}")

    # Using ascending frequency for all plots.
    if obs_obj.header['foff'] < 0:
        raw_psds = raw_psds[..., ::-1]  # Reverse data
        obs_freqs = obs_freqs[::-1] # Reverse frequencies

    print(f"revised obs_freqs {obs_freqs[0]} ... {obs_freqs[-1]}")

    # decimate in order to have a reasonable plot size
    minimum_effective_bw_mz = 10.0 # TODO we could calc this based on some file characteristic
    raw_effective_buckets = np.ceil(total_sampling_rate_mhz / minimum_effective_bw_mz)
    n_effective_buckets = round_to_nearest_power_of_two(raw_effective_buckets)
    print(f"raw_effective_buckets: {raw_effective_buckets} n_effective_buckets: {n_effective_buckets}")

    n_down_buckets = n_coarse_chan
    if n_down_buckets > n_effective_buckets:
        n_down_buckets = n_effective_buckets
    decimated_psds = None

    decimation_factor = int(len(raw_psds[0]) / n_down_buckets)
    plotted_freqs = obs_freqs[:-decimation_factor:decimation_factor]

    if decimation_factor > 1:
        print(f"Decimating {n_input_buckets} fine channels to {n_down_buckets} buckets, ratio: {n_input_buckets / n_down_buckets} ...")
        # perf_start_decimation = perf_counter()
        # decimated_psds = np.array([gaussian_decimate(row, n_down_buckets) for row in raw_psds])
        decimated_psds = raw_psds[::, ::decimation_factor] # faster but requires prior filtering
        print(f"decimated_psds shape: {decimated_psds.shape}")
        # print(f"Decimating >>> elapsed: {perf_counter()  - perf_start_decimation:0.3f} seconds")
    else:
        n_down_buckets = n_input_buckets
        decimated_psds = raw_psds

    # Calculate diffs between next and previous PSD (integration) rows
    # psd_diffs = decimated_psds[1:] - decimated_psds[:-1]
    psd_diffs = np.diff(decimated_psds, axis=0)
    # Compute the rate of change between time steps
    sec_per_integration = integration_period_sec / n_integrations_to_process
    dpsd_dt = psd_diffs / sec_per_integration
    print(f"dpsd_dt: {dpsd_dt.shape}")
    # rematch the input shape
    zero_row = np.zeros((1, dpsd_dt.shape[1]))
    dpsd_dt = np.vstack((zero_row, dpsd_dt))
    # scale rate of change relative to the PSD at that point
    dpsd_dt = np.divide(dpsd_dt, decimated_psds)

    # default_psd = mean_raw_psd
    # abs_dpsd_dt = np.abs(dpsd_dt)
    # threshold_frac = 0.6
    # thresh_dpsd_dt = np.max(abs_dpsd_dt) * threshold_frac
    # thresholded_psds = np.where(abs_dpsd_dt > thresh_dpsd_dt, decimated_psds, decimated_psds*(threshold_frac/2))

    plot_psds = decimated_psds
    # plot_psds = ndimage.maximum_filter(decimated_psds, size=6)
    # plot_psds = thresholded_psds

    plot_dpsd_dt = dpsd_dt
    # plot_dpsd_dt = ndimage.gaussian_filter1d(dpsd_dt, axis=0, sigma=3)

    print(f"shape psds: {plot_psds.shape} dpsd_dt: {plot_dpsd_dt.shape} ")

    perf_start_plotting = perf_counter()
    fig, axes = plt.subplots(nrows=2, figsize=full_screen_dims,  sharex=True, sharey=True, constrained_layout=True)
    fig.suptitle(f"{start_freq:0.4f} | {display_file_name} | N{n_down_buckets} | T{n_integrations_to_process}")

    cmap0 = matplotlib.colormaps['magma']
    cmap1 = matplotlib.colormaps['cividis']

    img0 = axes[0].imshow(plot_psds.T, aspect='auto', cmap=cmap0)
    img1 = axes[1].imshow(plot_dpsd_dt.T, aspect='auto', cmap=cmap1)
    axes[-1].set_xlabel('Timestep')

    cbar0 = fig.colorbar(img0, ax=axes[0])
    cbar0.set_label('Power dB', rotation=270, labelpad=15)
    cbar0.ax.yaxis.set_ticks_position('left')

    cbar1 = fig.colorbar(img1, ax=axes[1])
    cbar1.set_label('rate ΔdB/dt', rotation=270, labelpad=15)
    cbar1.ax.yaxis.set_ticks_position('left')

    orig_yticks = axes[0].get_yticks()
    print(f"axes0 orig_yticks ({len(orig_yticks)}) {orig_yticks}")
    orig_ylabels = axes[0].get_yticklabels()
    print(f"axes0 orig_ylabels ({len(orig_ylabels)}) {orig_ylabels}")

    print(f"plotted_freqs {plotted_freqs[0]} ... {plotted_freqs[-1]}")
    plotted_freq_range = plotted_freqs[-1] - plotted_freqs[0]
    print(f"plotted_freq_range: {plotted_freq_range}")

    tick_freqs = np.linspace(plotted_freqs[0], plotted_freqs[-1], num=len(orig_yticks))
    tick_step_mhz = tick_freqs[1] - tick_freqs[0]
    print(f"tick_freqs ({len(tick_freqs)}): {tick_freqs} tick_step_mhz: {tick_step_mhz}")
    # matplot places the first frequency offscreen, for whatever reason...
    tick_freqs -= tick_step_mhz
    print(f"tick_freqs ({len(tick_freqs)}): {tick_freqs}")
    ytick_labels = [f"{freq:0.3f}" for freq in tick_freqs]
    print(f"ytick_labels: {ytick_labels}")
    axes[0].set_yticklabels(ytick_labels,  rotation=30)
    axes[1].set_yticklabels(ytick_labels,  rotation=30)

    # TODO use a FixedLocator instead??
    # the following resets what actually gets plotted, rather than just the labels... 8^(
    # axes[0].set_yticks(ticks=orig_yticks, labels=ytick_labels)
    # axes[1].set_yticks(ticks=orig_yticks, labels=ytick_labels)
    # axes[2].set_yticks(ticks=orig_yticks, labels=ytick_labels)

    final_ylabels = axes[0].get_yticklabels()
    print(f"final_ylabels ({len(final_ylabels)}) : {final_ylabels}")
    print(f"Plotting >>> elapsed: {perf_counter()  - perf_start_plotting:0.3f} seconds")

    img_save_path = f"{plots_path}pow_diffs/{display_file_name}_N{n_down_buckets}_T{n_integrations_to_process}_dpsd.png"
    print(f"saving image to:\n{img_save_path}")
    plt.savefig(img_save_path)
    # plt.show()



if __name__ == "__main__":
    main()
