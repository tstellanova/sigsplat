"""
Example extracting various interesting info from HDF5 filterbank file
from a specific Voyager 1 observation:
http://blpd0.ssl.berkeley.edu/Voyager_data/Voyager1.single_coarse.fine_res.h5

Based on the tutorial at:
https://github.com/UCBerkeleySETI/blimpy/blob/master/examples/voyager.ipynb

"""
import argparse

import blimpy
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
from blimpy import Waterfall


matplotlib.use('qtagg')


def main():

    parser = argparse.ArgumentParser(description='Analyze HDF5 file')
    parser.add_argument('src_data_path', nargs='?',
                        help="Source hdf5 (.h5) file path",
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
    n_coarse_chan = obs_obj.calc_n_coarse_chan()
    print(f"n_coarse_chan: {n_coarse_chan}")

    print(f"data shape: {obs_obj.data.shape} , nbits: {int(obs_obj.header['nbits'])} , samples raw len: {len(obs_obj.data[0][0])}")
    one_sample = obs_obj.data[0,0,0]
    print(f"one_sample: {one_sample} dtype: {np.dtype(one_sample)} iscomplex: {np.iscomplex(one_sample)}")
    # two_sample = obs_obj.data[0,0,0]
    # print(f"two_sample: {two_sample} dtype: {np.dtype(two_sample)} iscomplex: {np.iscomplex(two_sample)}")

    number_of_integrations = obs_obj.data.shape[0]
    assert obs_obj.n_ints_in_file == number_of_integrations

    print(f"number_of_integrations {number_of_integrations}")
    # ctr: 8344.5 - 8419.30 =
    focus_freq_min = 8419.25 # 8419.29698
    focus_freq_max = 8419.35 #8419.2971
    if 'rawdatafile' in obs_obj.header:
        print(f"rawdatafile: {obs_obj.header['rawdatafile']}")
    print(f"plot PSD of data in the filterbank file (observation)")
    obs_obj.plot_spectrum(logged=True)
    plt.xticks(rotation=-45, ha="left")
    plt.show()
    # obs_obj.blank_dc(1)
    # obs_obj.plot_spectrum(logged=True)
    # plt.xticks(rotation=-45, ha="left")
    # plt.show()

    obs_obj.plot_spectrum(f_start=focus_freq_min, f_stop=focus_freq_max)
    # obs_obj.plot_spectrum(f_start=focus_freq_min, f_stop=focus_freq_max)
    plt.xticks(rotation=-45, ha="left")
    plt.show()

    # # obs_obj.plot_waterfall(f_start=8419.296, f_stop=8419.298)
    # # obs_obj.plot_waterfall(f_start=8419.29685, f_stop=8419.2971)
    # n_coarse_chan =  obs_obj.calc_n_coarse_chan()
    # freq_min = np.min(obs_obj.freqs)
    # freq_max = np.max(obs_obj.freqs)
    # freq_res = (freq_max - freq_min) / len(obs_obj.freqs)
    # ctr_freq = (freq_max + freq_min) / 2
    # stated_freqs = obs_obj.get_freqs()
    # print(f"freq_res: {freq_res} min: {freq_min} max: {freq_max} ctr: {ctr_freq}")
    # nchans = int(obs_obj.header['nchans'])
    # print(f"nchans (fine): {nchans} n_coarse_chan: {n_coarse_chan}")
    #
    # src_freqs, src_data = obs_obj.grab_data(f_start=focus_freq_min, f_stop=focus_freq_max)
    # if obs_obj.header['foff'] < 0:
    #     plot_data = src_data[..., ::-1]  # Reverse data
    #     plot_f = src_freqs[::-1]
    #
    # print(f"src_freqs len: {len(src_freqs)} src_data shape: {src_data.shape} ")
    #
    # plt.show()


if __name__ == "__main__":
    main()


