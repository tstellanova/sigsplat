"""
Examine filterbank ('.fil') or HDF5 ('.h5') files,
assumed to be from Breakthrough Listen archives,
and dump the header information.

A good place to obtain relevant open data:
http://seti.berkeley.edu/opendata

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
matplotlib.use('qtagg')

def main():

    parser = argparse.ArgumentParser(description='Analyze filterbank file')
    parser.add_argument('src_data_path', nargs='?',
                        help="Source hdf5 (.h5) or filerbank (.fil) file path",
                        # default="../../filterbank/blgcsurvey_cband/"
                        #     "spliced_blc00010203040506o7o0111213141516o7o0212223242526o7o031323334353637_guppi_58705_18741_BLGCsurvey_Cband_A00_0063.gpuspec.0002.fil"
                        default="../../filterbank/voyager1_rosetta_blc3/Voyager1.single_coarse.fine_res.h5"
                        )
    args = parser.parse_args()

    data_path = args.src_data_path
    print(f'loading file info: {data_path}')
    obs_obj = blimpy.Waterfall(data_path)
    # dump some interesting facts about this observation
    print(">>> Dump observation info...")
    obs_obj.info()
    print(f">>> observation header: {obs_obj.header} ")

    if 'rawdatafile' in obs_obj.header:
        print(f"Source raw data file: {obs_obj.header['rawdatafile']}")
    else:
        print(f"No upstream raw data file reported in header")

    # the following reads through the filterbank file in order to calculate the number of coarse channels
    n_coarse_chan = int(obs_obj.calc_n_coarse_chan())
    n_fine_chan = obs_obj.header['nchans']

    fine_channel_bw_mhz = obs_obj.header['foff']
    fine_channel_bw_hz = np.abs(fine_channel_bw_mhz) * 1E6
    print(f"n_coarse_chan: {n_coarse_chan} n_fine_chan: {n_fine_chan} fine_channel_bw_hz: {fine_channel_bw_hz}")
    n_integrations_input = obs_obj.n_ints_in_file
    print(f"Num integrations in file: {n_integrations_input}")
    n_polarities_stored = obs_obj.header['nifs']
    print(f"n_polarities_stored {n_polarities_stored}")

    print(f"data shape: {obs_obj.data.shape} , nbits: {int(obs_obj.header['nbits'])} , freqs per integration: {len(obs_obj.data[0][0])}")

    # validate that the input file data shape matches expectations set by header
    # TODO this all explodes for very large data files where the Waterfall constructor can't load all the data
    assert n_integrations_input == obs_obj.data.shape[0]
    assert n_polarities_stored == obs_obj.data.shape[1]
    assert n_fine_chan == obs_obj.data.shape[2]


    # tsamp is "Time integration sampling rate in seconds" (from rawspec)
    integration_period_sec = float(obs_obj.header['tsamp'])
    print(f"Integration interval: {integration_period_sec} seconds")
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
    print(f"one_sample dtype: {np.dtype(one_sample)} iscomplex: {np.iscomplex(one_sample)}")
    # two_sample = obs_obj.data[0,0,0]
    # print(f"two_sample: {two_sample} dtype: {np.dtype(two_sample)} iscomplex: {np.iscomplex(two_sample)}")
    fig = plt.figure(figsize=(16,10), constrained_layout=True)
    # plot one spectrum
    obs_obj.plot_spectrum(logged=True, t=n_integrations_input//2)

    obs_obj.plot_waterfall()

    # overlay one fine line per time step
    ax0 = fig.get_axes()[0]
    y_bott, y_top = ax0.get_ylim()
    cmap0 = matplotlib.colormaps['viridis']
    timescale_factor =  (y_top - y_bott) / n_integrations_input
    for time_step in range(n_integrations_input):
        ax0.axhline(y=timescale_factor*time_step, xmin=0, xmax=25, color=cmap0(0.5))

    plt.show()


if __name__ == "__main__":
    main()
