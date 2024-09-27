
from abc import abstractmethod

import numpy as np
import logging
import sys

# fount_logger = logging.getLogger(__name__)
# level_log = logging.INFO
#
# if level_log == logging.INFO:
#     stream = sys.stdout
#     lformat = '%(name)-15s %(levelname)-8s %(message)s'
# else:
#     stream =  sys.stderr
#     lformat = '%%(relativeCreated)5d (name)-15s %(levelname)-8s %(message)s'
#
# logging.basicConfig(format=lformat, stream=stream, level=level_log)


class DataFount(object):
    """
    A fountain of data
    """
    def __init__(self, num_polarizations:int = 1, loglevel = logging.WARN):
        super(DataFount, self).__init__()
        self.n_pols = num_polarizations
        self.logger = self.create_logger(loglevel)

    def create_logger(self, loglevel):
        display_name = type(self).__name__
        fount_logger = logging.getLogger(display_name)

        if loglevel == logging.INFO:
            stream = sys.stdout
            lformat = '%(name)-15s %(levelname)-8s %(message)s'
        else:
            stream =  sys.stderr
            lformat = '%%(relativeCreated)5d (name)-15s %(levelname)-8s %(message)s'

        logging.basicConfig(format=lformat, stream=stream, level=loglevel)
        fount_logger.info(f"datafount log created")

        return fount_logger

    @abstractmethod
    async def read_samples(self, n_samples:int = 64) -> np.ndarray[np.complex64] :
        pass



