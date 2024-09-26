import os

import numpy
import numpy as np
from sigsplat.founts.base_fount import DataFount


class RawBasebandFileFount(DataFount):
    """ This class handles raw baseband files
    """

    def __init__(self, filepath:str, is_complex:bool=True, in_dtype:np.dtype=np.int8):
        super(RawBasebandFileFount, self).__init__()
        self.logger.info(f"datafount {__name__} init")
        if filepath and os.path.isfile(filepath):
            self.file_size_bytes = os.path.getsize(filepath)
            self.file_dtype = in_dtype
            self.file_is_complex = is_complex

            self.sample_size_bytes = np.dtype(in_dtype).itemsize
            if self.file_is_complex:
                self.sample_size_bytes *= 2
            self.infile = open(filepath, mode='rb')
        else:
            raise IOError(f"Fount unable to open file at {filepath}")

    async def read_samples(self, n_samples: int = 64) -> np.ndarray[np.complex64]:
        # Asynchronously read a chunk of data from the file
        bytes_to_read = n_samples * self.sample_size_bytes
        raw_data = self.infile.read(bytes_to_read)

        if not raw_data:
            return None  # End of file

        # TODO should we file.read multiple times if we can't obtain what we want in one shot?
        # Convert the raw bytes into a NumPy array of the specified dtype
        complex_array = np.frombuffer(raw_data, dtype=np.complex64)

        return complex_array



# async def process_iq_samples(stream_reader, chunk_size=1024):
#     while True:
#         # Asynchronously read a chunk of data from the stream (non-blocking)
#         raw_data = await stream_reader.read(chunk_size)
#
#         if not raw_data:
#             break  # End of stream
#
#         # Assuming the raw_data is bytes, convert to complex64 (I/Q samples)
#         iq_samples = np.frombuffer(raw_data, dtype=np.complex64)
#
#         # Process the I/Q samples here
#         await process_samples(iq_samples)

