import sys
import numpy as np
from scipy import ndimage
from scipy.linalg import eigh
from time import perf_counter


def round_to_nearest_power_of_two(n):
    return int(np.power(2, np.round(np.log2(n))))


def normalize_data(data: np.ndarray) -> np.ndarray:
    """
    Rescale the data to be small +/-
    :param data:
    :return: Data rescaled to be between about -1.0 and 1.0
    """
    return (data - np.mean(data)) / np.std(data)


def safe_scale_log(data: np.ndarray) -> np.ndarray:
    """
    Take the logarithm of the absolute value of the input, then reattach the sign.
    Zero values are acceptable
    :param data:
    :return: Data on a +/- log scale
    """
    return 10 * np.sign(data) * np.log10(np.abs(data) + sys.float_info.epsilon)


def simple_decimate(data, num_out_buckets: int = None, decimation_factor: int = None) -> np.ndarray:
    """
    A fast but dumb function that doens't perform any pre-filtering on the data before decimating
    :param data:
    :param num_out_buckets:
    :return:
    """
    if num_out_buckets is not None:
        decimation_factor = len(data) // num_out_buckets
    decimated_data = data[::decimation_factor]
    return decimated_data


def gaussian_decimate(data: np.ndarray, num_out_buckets: int = 256, sigma=1.0) -> np.ndarray:
    """
    Decimate a big ndarray to a smaller array, smoothing input data first.
    See also:   scipy.signal.decimate()

    :param data: Input ndarray (1d)
    :param num_out_buckets: The desired length of the output
    :param sigma: Sigma to use for gaussian filtering
    :return: Filtered (smoothed) and decimated 1d
    """
    decimation_factor = len(data) // num_out_buckets

    if decimation_factor <= 1:
        print(f"Not decimating {len(data)} to {num_out_buckets}")
        return data

    # Apply a smoothing filter across the data
    filtered_data = ndimage.gaussian_filter1d(data, sigma=sigma)

    # Decimate (downsample)
    decimated_data = filtered_data[::decimation_factor]
    return decimated_data


def adaptive_filter_dual_pol(pol_a: np.ndarray, pol_b: np.ndarray) -> np.ndarray:
    """
    Given simultaneous samples from two (presumably orthogonal) polarities,
    use Adaptive Filtering to first find the covariance between the two polarities,
    then return the principal signal that is common to both.

    :param pol_a:
    :param pol_b:
    :return:
    """
    outshape = pol_a.shape
    out_dtype = pol_a.dtype
    # print(f"Start Adaptive Filter...for {outshape} ({out_dtype})")

    # perf_start_adapt_filt = perf_counter()
    # Form the 2xN matrix from the two polarizations (N samples per polarization)
    X = np.vstack((pol_a.flatten(), pol_b.flatten()))
    # print(f"stacked pols: {X.shape} vs pol_a: {pol_a.shape}")

    # Compute the covariance matrix of the signals
    R = np.cov(X)  # This gives a 2x2 covariance matrix
    # print(f"covariance shape: {R.shape}\n{R}")

    # Perform eigenvalue decomposition on the covariance matrix
    eigenvalues, eigenvectors = eigh(R)
    # print(f"eigenvalues: {eigenvalues} \neigenvectors:\n{eigenvectors}")
    # Select the eigenvector associated with the largest eigenvalue (this is the principal component)
    peak_eigenvec = np.argmax(eigenvalues)
    # print(f"peak_eigenvec: {peak_eigenvec}")
    principal_component = eigenvectors[:, peak_eigenvec]

    # Filter the signals using the principal component (project onto the dominant direction)
    principal_signal = np.dot(principal_component, X)
    # print(f"reshape {principal_signal.shape} into {outshape}")
    principal_signal = np.reshape(principal_signal, outshape)
    principal_signal = np.astype(principal_signal, out_dtype)
    # print(f"Adapt Filter >>> elapsed: {perf_counter()  - perf_start_adapt_filt:0.3f} seconds")
    return principal_signal


def remove_extreme_power_peaks(arr_db: np.ndarray, db_threshold=5) -> np.ndarray:
    """
    Remove extreme peaks from an array based on their exceptional power
    :param arr_db: Input power data, assumed to be in a decibel (log10) format
    :param db_threshold:
    :return:
    """

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


# TODO canonical method for STFT processing of ongoing raw sample updates
