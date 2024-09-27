import logging
import random

import numpy as np
from numpy.random import default_rng

from sigsplat.founts.base_fount import DataFount


class WgnFount(DataFount):
    """
    This class generates White Gaussian Noise samples...forever
    """

    def __init__(self, num_polarizations: int = 1, seed_hint=None):
        super(WgnFount, self).__init__(num_polarizations=num_polarizations, loglevel=logging.INFO)
        self.rng = default_rng(seed=seed_hint)

    async def read_samples(self, n_samples: int = 64) -> np.ndarray[np.complex64]:
        result_shape = (self.n_pols, n_samples)
        result = np.empty(result_shape, dtype=np.complex64)
        for pol_idx in range(self.n_pols):
            # Generate PRN real and imaginary parts
            real_sig = 2*self.rng.random(n_samples, dtype=np.float32) - 1.
            im_sig = 2*self.rng.random(n_samples, dtype=np.float32) - 1.
            # Combine them into complex64 array
            complex_samples = (real_sig + 1j * im_sig).astype(np.complex64)
            result[pol_idx] = complex_samples

        # self.logger.info(f"pre-squeeze: {result.shape} {result.dtype} ")
        result = result.squeeze() # if we have a (1, 5048) ndarray, squeeze down to a single row
        # self.logger.info(f"post-squeeze: {result.shape} {result.dtype} ")
        return result
