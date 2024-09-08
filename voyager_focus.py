"""
Example used to extract the Voyager narrowband signal
from a GUPPI .raw file
as well as from a filterbank .fil file
This example assumes certain well-known GBT files are present in a ./data subdirectory
"""
import blimpy.io.sigproc
import pylab as plt
from pprint import pprint
from blimpy import Waterfall, GuppiRaw
import numpy as np
import matplotlib
# this might only be necessary on macos: run eg 'pip install PyQt5' first
matplotlib.use('qtagg')

obs = Waterfall('../../filterbank/misc/voyager_f1032192_t300_v2.fil')
obs.info()
# this the data printed by obs.info:
pprint(obs.header)
print("obs data shape: ", obs.data.shape)

# print("n_coarse_chan: ", obs.calc_n_coarse_chan(2.94))
# r = GuppiRaw('./data/blc03_samples/blc3_guppi_57386_VOYAGER1_0004.0000.raw')
# first_header = r.read_first_header()
# pprint(first_header)
# r.find_n_data_blocks()
# r.read_next_data_block_shape()
# r.plot_spectrum()
#r.plot_histogram()

my_dpi=200
full_screen_dims=(3456 / my_dpi, 2234 / my_dpi)
# plot the entire power spectrum collected
#plt.figure(figsize=full_screen_dims, dpi=my_dpi)
#obs.plot_spectrum()
#plt.xticks(rotation=-45, ha="left")
#plt.title("full spectrum")
#plt.savefig('./img/spectrum.png')

plt.figure(figsize=full_screen_dims, dpi=my_dpi)
obs.plot_waterfall(f_start=8420.193, f_stop=8420.24, logged=True)
plt.title("waterfall")
plt.savefig('./img/waterfall.png')

plt.figure(figsize=full_screen_dims)
plt.xticks(rotation=-45, ha="left")
obs.plot_spectrum(f_start=8420.18, f_stop=8420.26, logged=True) #from sideband to sideband
plt.title("full signal")
plt.savefig('./img/full_sig.png')

# todo measure full range
(plot_f, plot_efields) = obs.grab_data(f_start=8402.0, f_stop=8588.0)
# print("frequencies ", plot_f.shape)
# pprint(plot_f)
# print("efields ", plot_efields.shape)
# pprint(plot_efields)

# calculate magnitude of difference
pol_delta = (plot_efields[1] - plot_efields[0]) ** 2
pol_mag = np.sqrt(pol_delta)
# print("pol_mag ", pol_mag.shape)
# pprint(pol_mag)

# plt.figure(figsize=full_screen_dims, dpi=my_dpi)
# plt.title("timediff")
# plt.ylabel("mag(diff)")
# plt.xlabel("frequency")
# plt.plot(plot_f, pol_mag)

# plt.plot(plot_f, plot_efields[0])
# plt.legend()
# plt.savefig('timediff.png')

# plot a narrow area around frequency of interest
plt.figure(figsize=full_screen_dims)
# fig, axes = plt.subplots(nrows=3, figsize=full_screen_dims,  sharex=True)
plt.subplot(3,1,1)
plt.xticks(rotation=-45, ha="left")
obs.plot_spectrum(f_start=8420.193, f_stop=8420.195) # left sideband
plt.title("left sideband")
plt.subplot(3,1,2)
plt.xticks(rotation=-45, ha="left")
obs.plot_spectrum(f_start=8420.2163, f_stop=8420.2166) # carrier
plt.title("carrier")
plt.subplot(3,1,3)
plt.xticks(rotation=-45, ha="left")
obs.plot_spectrum(f_start=8420.238, f_stop=8420.24) # right sideband
plt.title("right sideband")
plt.tight_layout()
plt.savefig('./img/bands.png')

# plt.figure(figsize=full_screen_dims)
# plt.xticks(rotation=-45, ha="left")
# obs.plot_time_series()
# plt.savefig('time_series.png')

# keep plots on screen until process is killed
plt.show()
