import numpy as np
from blimpy import Waterfall
from time import perf_counter

from sigsplat.convert import spectralize
from scipy.ndimage import gaussian_filter1d


def grab_next_integration(obs_obj: Waterfall) -> (np.ndarray, np.ndarray):
    cur_freqs, cur_ps = obs_obj.grab_data()
    if cur_ps.dtype != np.float32:
        # TODO should we rescale these, or does it not matter?
        cur_ps = np.float32(cur_ps)
    return cur_freqs, cur_ps


def grab_one_integration(obs_obj: Waterfall, int_idx=0) -> (np.ndarray, np.ndarray):
    n_ints_in_file = int(obs_obj.n_ints_in_file)
    if int_idx > n_ints_in_file:
        print(f"Exceeded integrations in file: {n_ints_in_file}")
        return None
    n_avail_ints = obs_obj.data.shape[0]
    if int_idx > n_avail_ints:
        obs_obj.read_data(t_start=int_idx, t_stop=int_idx + 1)
    return grab_next_integration(obs_obj)


def grab_desired_spectra(obs_obj: Waterfall, n_integrations: int, n_input_buckets: int) -> (np.ndarray, np.ndarray):
    raw_psds = np.zeros((n_integrations, n_input_buckets), dtype=np.float32)

    cur_freqs = None  # TODO roll up all observed frequencies
    print(f"Collecting {n_integrations} integrations ...")
    perf_start_collection = perf_counter()
    for int_idx in range(n_integrations):
        cur_freqs, cur_ps = grab_one_integration(obs_obj, int_idx)
        # convert power counts to db
        cur_ps = spectralize.safe_scale_log(cur_ps)
        # The filt data often has one or more narrow "DC-offset" peaks -- we attempt to remove those here
        # cur_ps = spectralize.remove_extreme_power_peaks(cur_ps, db_threshold=3)
        # this performs a low-pass filter on changes between adjacent frequency bins
        cur_ps = gaussian_filter1d(cur_ps, sigma=3)
        raw_psds[int_idx] = cur_ps
    print(f"Collection >>> elapsed: {perf_counter() - perf_start_collection:0.3f} seconds")

    min_raw_psd = np.min(raw_psds)
    max_raw_psd = np.max(raw_psds)

    print(f"Raw PSDs range: {min_raw_psd:0.3e} ... {max_raw_psd:0.3e}")

    return cur_freqs, raw_psds
