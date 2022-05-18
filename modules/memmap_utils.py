import sys
import ctypes
import logging


def init_madvise(memmap):
    if 'win' not in sys.platform:
        try:
            madvise = ctypes.CDLL("libc.so.6").madvise
            madvise.argtypes = [ctypes.c_void_p, ctypes.c_size_t, ctypes.c_int]
            madvise.restype = ctypes.c_int
            assert madvise(memmap.ctypes.data, memmap.size * memmap.dtype.itemsize, 1) == 0, "MADVISE FAILED" # 1 means MADV_RANDOM
        except:
            logging.info(f'MADVISE FAILED')
