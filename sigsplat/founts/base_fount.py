
import asyncio
from abc import abstractmethod

import numpy as np
import logging
import sys

fount_logger = logging.getLogger(__name__)
level_log = logging.INFO

if level_log == logging.INFO:
    stream = sys.stdout
    lformat = '%(name)-15s %(levelname)-8s %(message)s'
else:
    stream =  sys.stderr
    lformat = '%%(relativeCreated)5d (name)-15s %(levelname)-8s %(message)s'

logging.basicConfig(format=lformat, stream=stream, level=level_log)


class DataFount(object):
    """
    A fountain of data
    """
    def __init__(self):
        super(DataFount, self).__init__()
        self.logger = fount_logger

    @abstractmethod
    async def read_samples(self, n_samples:int=64) -> np.ndarray[np.complex64] :
        pass



