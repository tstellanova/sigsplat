import matplotlib.pyplot as plt
import numpy as np

from sigsplat.convert import spectralize


def plot_compared_psd(psds_a: np.ndarray, psds_b: np.ndarray, suptitle: None, show: bool = False,
                      img_save_path=None):

    print(f"psds_a {psds_a.shape} {psds_a.dtype} ")
    print(f"psds_b {psds_b.shape} {psds_b.dtype}")
    # TODO allow setting freqs list
    full_screen_dims = (16, 10)
    fig, axes = plt.subplots(nrows=2, figsize=full_screen_dims, constrained_layout=True, sharex=True, sharey=True)
    if suptitle is not None:
        fig.suptitle(suptitle)
    img0 = axes[0].imshow(psds_a, aspect='auto', cmap='inferno')
    img1 = axes[1].imshow(psds_b, aspect='auto', cmap='viridis')
    axes[-1].set_xlabel('Time')

    cbar0 = fig.colorbar(img0, ax=axes[0])
    cbar0.set_label('PSD (dbFS)', rotation=270, labelpad=15)
    cbar0.ax.yaxis.set_ticks_position('left')
    cbar1 = fig.colorbar(img1, ax=axes[1])
    cbar1.set_label('PSD (dbFS)', rotation=270, labelpad=15)
    cbar1.ax.yaxis.set_ticks_position('left')

    if img_save_path is not None:
        print(f"saving plot to:\n{img_save_path} ...")
        plt.savefig(img_save_path)

    if show:
        plt.show()


def plot_compared_bb_psd(signal_a: np.ndarray, signal_b: np.ndarray, n_freq_bins: int = 64,
                         sampling_freq_hz: np.float32 = 1E6, suptitle=None, show: bool = False, img_save_path=None):
    stft_obj = None
    stft_obj, psd_series_a = spectralize.calc_psd_using_stft(signal_a, stft_obj=stft_obj, n_freq_bins=n_freq_bins,
                                                             sampling_freq_hz=sampling_freq_hz)
    stft_obj, psd_series_b = spectralize.calc_psd_using_stft(signal_b, stft_obj=stft_obj, n_freq_bins=n_freq_bins,
                                                             sampling_freq_hz=sampling_freq_hz)


    plot_compared_psd(psd_series_a, psd_series_b, show=show, suptitle=suptitle, img_save_path=img_save_path)
