import asyncio

import numpy as np

import sigsplat
from sigsplat.founts.raw_file_fount import RawBasebandFileFount
from sigsplat.founts.wgn_fount import WgnFount


def main():
    wgn_fount = WgnFount(555)
    for i in range(100):
        le_samples = asyncio.run(wgn_fount.read_samples(32))
        print(f"{i} read {le_samples.shape} {le_samples.dtype} ")
    wgn_fount = None

    sample_file_path = "../../../baseband/wbfm/gqrx_20240924_190900_90699998_28000000_fc.raw"
    raw_file_fount = RawBasebandFileFount(sample_file_path, is_complex=True, in_dtype=np.float32)
    while True:
        n_read = 0
        le_samples = asyncio.run(raw_file_fount.read_samples(int(1E9)))
        if le_samples is not None:
            n_read = len(le_samples)
        if n_read > 0:
            print(f"{i} nread: {n_read} :  {le_samples.shape} {le_samples.dtype} ")
        if n_read < 1E9:
            break

if __name__ == "__main__":
    main()

