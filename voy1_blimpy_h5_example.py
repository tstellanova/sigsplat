"""
Example extracting various interesting info from HDF5 filterbank file
from a specific Voyager 1 observation:
http://blpd0.ssl.berkeley.edu/Voyager_data/Voyager1.single_coarse.fine_res.h5
"""
import argparse

import blimpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import sigmf
from matplotlib.ticker import StrMethodFormatter
from sigmf import sigmffile, SigMFFile
from blimpy import Waterfall


matplotlib.use('qtagg')



def main():

    parser = argparse.ArgumentParser(description='Analyze HDF5 file')
    parser.add_argument('--src_data_path',
                        help="Source hdf5 (.h5) file path",
                        default="./data/Voyager1.single_coarse.fine_res.h5")
    # parser.add_argument('--min_lag_ms',dest='min_lag_ms', type=float, default=4.0,
    #                     help="Minimum repeat time in ms (for autocorrelation calculation)")
    args = parser.parse_args()

    data_path = args.src_data_path
    print(f'loading file info: {data_path}')
    obs_obj = blimpy.Waterfall(data_path)
    # dump some interesting facts about this observation
    obs_obj.info()
    print(obs_obj.header)
    print(f"data shape: {obs_obj.data.shape} nbits: {int(obs_obj.header['nbits'])} samples raw len: {len(obs_obj.data[0][0])}")

    focus_freq_min = 8419.29698
    focus_freq_max = 8419.2971
    print(f"rawdatafile: {obs_obj.header['rawdatafile']}")
    print(f"plot PSD of data in the filterbank file (observation)")
    obs_obj.plot_spectrum(logged=True)
    plt.show()
    obs_obj.blank_dc(1)
    obs_obj.plot_spectrum(logged=True)
    plt.show()
# obs_obj.plot_spectrum(f_start=8419.26, f_stop=8419.34)
    obs_obj.plot_spectrum(f_start=focus_freq_min, f_stop=focus_freq_max)
    plt.xticks(rotation=-45, ha="left")
    # plt.show()

    # obs_obj.plot_waterfall(f_start=8419.296, f_stop=8419.298)
    # obs_obj.plot_waterfall(f_start=8419.29685, f_stop=8419.2971)
    n_coarse_chan =  obs_obj.calc_n_coarse_chan()
    # obs_obj.blank_dc(n_coarse_chan)
    # replot spectrum after DC blanking
    # obs_obj.plot_spectrum(f_start=8419.29685, f_stop=8419.2971)

    freq_min = np.min(obs_obj.freqs)
    freq_max = np.max(obs_obj.freqs)
    freq_res = (freq_max - freq_min) / len(obs_obj.freqs)
    ctr_freq = (freq_max + freq_min) / 2
    print(f"freq_res: {freq_res} min: {freq_min} max: {freq_max} ctr: {ctr_freq}")
    nchans = int(obs_obj.header['nchans'])
    print(f"nchans: {nchans} n_coarse_chan: {n_coarse_chan}")

    src_freqs, src_data = obs_obj.grab_data(f_start=focus_freq_min, f_stop=focus_freq_max)
    if obs_obj.header['foff'] < 0:
        plot_data = src_data[..., ::-1]  # Reverse data
        plot_f = src_freqs[::-1]

    print(f"src_freqs len: {len(src_freqs)} src_data shape: {src_data.shape} ")

    plt.show()








if __name__ == "__main__":
    main()


