"""Optional drop-behind for already copied, read-only mmap input.

This is an advisory cache policy, not a memory limit or a data deletion. Only
whole pages contained in the consumed contiguous view are eligible. Ordinary
arrays, writable/COW maps and strided views are left alone.
"""
import mmap
import os
import stat

import numpy as np


def release_consumed_input(values):
    result = {"eligible_bytes": 0, "advised_bytes": 0, "error": None}
    if not (hasattr(os, "posix_fadvise") and hasattr(mmap.mmap, "madvise")
            and hasattr(mmap, "MADV_DONTNEED")):
        return result
    array = np.asarray(values)
    if not array.flags.c_contiguous or not array.nbytes:
        return result
    root = array
    mapping = None
    while isinstance(root, np.ndarray):
        if isinstance(root, np.memmap):
            mapping = root
        root = root.base
    if mapping is None or mapping.mode != "r" or not isinstance(root, mmap.mmap):
        return result
    delta = array.ctypes.data - mapping.ctypes.data
    if delta < 0 or delta + array.nbytes > mapping.nbytes:
        return result
    page = mmap.PAGESIZE
    lo = ((mapping.offset + delta + page - 1) // page) * page
    hi = ((mapping.offset + delta + array.nbytes) // page) * page
    if hi <= lo:
        return result
    result["eligible_bytes"] = hi - lo
    # numpy's mmap starts at an allocation-granularity boundary, not at the
    # array's possibly unaligned file offset (e.g. an eight-byte fbin header).
    map_start = mapping.offset - mapping.offset % mmap.ALLOCATIONGRANULARITY
    fd = None
    try:
        fd = os.open(mapping.filename, os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0))
        info = os.fstat(fd)
        if not stat.S_ISREG(info.st_mode) or hi > info.st_size:
            return result
        # Drop our PTEs first: fadvise alone may leave mmap-referenced pages.
        # Both calls preserve file bytes and permit later rereads.
        root.madvise(mmap.MADV_DONTNEED, lo - map_start, hi - lo)
        os.posix_fadvise(fd, lo, hi - lo, os.POSIX_FADV_DONTNEED)
        result["advised_bytes"] = hi - lo
    except (OSError, ValueError) as exc:
        # An unsupported advisory must not turn a committed insert into a
        # failed benchmark or silently pretend that memory was released.
        result["error"] = str(exc)
    finally:
        if fd is not None:
            os.close(fd)
    return result
