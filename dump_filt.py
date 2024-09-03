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
# import matplotlib.pyplot as plt

def main():

    parser = argparse.ArgumentParser(description='Analyze HDF5 file')
    parser.add_argument('--src_data_path',
                        help="Source hdf5 (.h5) file path",
                        # default="./data/voyager1_rosetta_blc3/Voyager1.single_coarse.fine_res.h5"
                        default="./data/voyager1_rosetta_blc3/Voyager1.single_coarse.fine_res.fil"
                        # default="./data/blc07_samples/blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.fil"
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
    print(f"n_integrations_input {n_integrations_input}")

    print(f"data shape: {obs_obj.data.shape} , nbits: {int(obs_obj.header['nbits'])} , freqs per integration: {len(obs_obj.data[0][0])}")

    number_of_integrations = obs_obj.data.shape[0]
    assert number_of_integrations == n_integrations_input

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



if __name__ == "__main__":
    main()
