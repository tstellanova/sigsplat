import random

import numpy as np
from numpy.random import default_rng

from sigsplat.founts.base_fount import DataFount


class WgnFount(DataFount):
    """
    This class generates White Gaussian Noise samples...forever
    """

    def __init__(self, seed_hint=None):
        super(WgnFount, self).__init__()
        self.logger.info(f"datafount {__name__} init")
        self.rng = default_rng(seed=seed_hint)

    async def read_samples(self, n_samples:int = 64) -> np.ndarray[np.complex64]:
        # Generate PRN real and imaginary parts
        real_part = self.rng.random(n_samples).astype(np.float32)
        imaginary_part = self.rng.random(n_samples).astype(np.float32)

        # Combine them into complex64 array
        complex_array = (real_part + 1j * imaginary_part).astype(np.complex64)

        return complex_array
