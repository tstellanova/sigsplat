import asyncio

import numpy as np

from sigsplat.convert import spectralize
from sigsplat.founts.wgn_fount import WgnFount
from sigsplat.plotting import spectra_plots
import matplotlib
matplotlib.use('qtagg')

def main():
    wgn_fount = WgnFount(seed_hint=555, num_polarizations=2)
    n_time_steps = 100
    n_samples_per_time_step = 64
    voltage_shape = (n_time_steps, n_samples_per_time_step)
    raw_sig = np.empty(voltage_shape, dtype=np.complex64)
    denoised_sig = np.empty(voltage_shape, dtype=np.complex64)
    for time_idx in range(n_time_steps):
        le_samples = asyncio.run(wgn_fount.read_samples(n_samples_per_time_step))
        pol_a = le_samples[0, ...]
        pol_b = le_samples[1, ...]
        denoised_sig[time_idx] = spectralize.adaptive_filter_dual_pol(pol_a, pol_b)
        raw_sig[time_idx] = pol_a

    print(f"raw_sig {raw_sig.shape} {raw_sig.dtype}")
    print(f"denoised_sig {denoised_sig.shape} {denoised_sig.dtype}")

    spectra_plots.plot_compared_bb_psd(raw_sig.flatten(),
                                       denoised_sig.flatten(),
                                       n_freq_bins=256,
                                       sampling_freq_hz=10E6,
                                       suptitle='This Acid, Man',
                                       show=True)


if __name__ == "__main__":
    main()
