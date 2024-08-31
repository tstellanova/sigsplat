import numpy as np
import matplotlib.pyplot as plt
import argparse
import re
import os
import blimpy

import matplotlib
matplotlib.use('qtagg')

def compute_hog_1d(signal, cell_size=10, num_bins=9):
    """
    Compute the HOG for a 1D signal.

    Parameters:
    signal (np.ndarray): The 1D array of integrated power spectra.
    cell_size (int): The size of each cell.
    num_bins (int): The number of bins for the histogram.

    Returns:
    np.ndarray: The histogram of gradients for the signal.
    """
    # Compute gradients
    gradients = np.diff(signal)

    # Compute the angles and magnitudes of the gradients
    magnitudes = np.abs(gradients)
    angles = np.sign(gradients)  # In 1D, angles are either -1 or 1

    # Divide the signal into cells
    num_cells = len(signal) // cell_size
    hog = np.zeros((num_cells, num_bins))

    # Compute the histogram for each cell
    for i in range(num_cells):
        start = i * cell_size
        end = start + cell_size
        cell_magnitudes = magnitudes[start:end]
        cell_angles = angles[start:end]

        # Create histogram for the cell
        bin_edges = np.linspace(-1, 1, num_bins + 1)
        hist, _ = np.histogram(cell_angles, bins=bin_edges, weights=cell_magnitudes)
        hog[i] = hist

    return hog

def plot_hog_time_series(hog_series, cell_size, time_labels):
    """
    Plot the HOG for a time series of spectra.

    Parameters:
    hog_series (np.ndarray): The HOG for each spectrum in the time series (shape: num_spectra x num_cells x num_bins).
    cell_size (int): The size of each cell.
    time_labels (list): The labels for the time axis.
    """
    num_spectra, num_cells, num_bins = hog_series.shape
    fig, ax = plt.subplots(figsize=(18, 9))

    for t in range(num_spectra):
        for i in range(num_cells):
            for j in range(num_bins):
                height = hog_series[t, i, j]
                ax.plot([i * cell_size, (i + 1) * cell_size], [j + t * num_bins, j + t * num_bins], color='black', linewidth=height)

    ax.set_title("HOG for Time Series of Integrated Power Spectra")
    ax.set_xlabel("Frequency Bins")
    ax.set_ylabel("Time and Gradient Direction Bin")
    ax.set_yticks(np.arange(0, num_spectra * num_bins, num_bins) + num_bins / 2)
    ax.set_yticklabels(time_labels)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Use HOG technique on integrated power spectra')
    parser.add_argument('--src_data_path',
                        help="Source filterbank (.fil or .h5) file path",
                        default="./data/voyager1_rosetta_blc3/Voyager1.single_coarse.fine_res.h5"
                        # default="./data/voyager1_rosetta_blc3/Voyager1.single_coarse.fine_res.fil"
                        # default="./data/blc07_samples/blc07_guppi_57650_67573_Voyager1_0002.gpuspec.0000.fil"
                        )
    parser.add_argument("--out_path",dest='out_path',default='./',
                        help="Directory path to place output files" )

    args = parser.parse_args()
    data_path = args.src_data_path
    out_path = args.out_path

    print(f'loading file info: {data_path}')
    obs_obj = blimpy.Waterfall(data_path)
    print(">>> Dump observation info...")
    obs_obj.info()

    n_coarse_chan = obs_obj.calc_n_coarse_chan()
    print(f"n_coarse_chan: {n_coarse_chan}")

    print(f"data shape: {obs_obj.data.shape} , nbits: {int(obs_obj.header['nbits'])} , samples raw len: {len(obs_obj.data[0][0])}")

    n_ints_in_file = obs_obj.n_ints_in_file
    print(f"n_ints_in_file: {n_ints_in_file}")
    fine_freqs = obs_obj.get_freqs()
    print(f"fine_freqs: {fine_freqs.shape}")

    plot_f, plot_data = obs_obj.grab_data()
    print(f"plot_f shape: {plot_f.shape}")
    print(f"plot_data shape: {plot_data.shape}")

    # n_coarse_chan: 1.0
    # data shape: (16, 1, 1048576) , nbits: 32 , samples raw len: 1048576
    # number_of_integrations 16

    n_integrations_to_process = 2
    n_cells_per_spectra = 64
    cell_size = len(fine_freqs) // n_cells_per_spectra
    num_hist_bins = 100

    # Compute the HOG for each integrated power spectrum
    hog_series = np.zeros((n_integrations_to_process, n_cells_per_spectra, num_hist_bins))
    for int_idx in range(n_integrations_to_process):
        spectrum = plot_data[int_idx]
        hog_series[int_idx] = compute_hog_1d(spectrum, cell_size=cell_size, num_bins=num_hist_bins)

    # Create time labels
    time_labels = [f"t={i}" for i in range(n_integrations_to_process)]

    # Plot the HOG for the time series
    plot_hog_time_series(hog_series, cell_size=cell_size, time_labels=time_labels)

if __name__ == "__main__":
    main()