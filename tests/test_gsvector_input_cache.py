import hashlib
import mmap
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from neurips23.streaming.gsvector.input_cache import release_consumed_input


@unittest.skipUnless(hasattr(os, "posix_fadvise") and hasattr(mmap.mmap, "madvise"),
                     "requires mmap and file-cache advice")
class InputCacheTests(unittest.TestCase):
    def fixture(self, directory, mode="r", offset=8):
        path = Path(directory) / "vectors.bin"
        data = np.arange(40000, dtype="float32").reshape(400, 100)
        path.write_bytes(b"x" * offset + data.tobytes())
        return path, data, np.memmap(path, mode=mode, dtype="float32",
                                     offset=offset, shape=data.shape)

    def test_only_whole_consumed_pages_and_reread_is_identical(self):
        with tempfile.TemporaryDirectory() as directory:
            path, data, values = self.fixture(directory)
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            view = np.asarray(values[21:143])
            with patch("os.posix_fadvise", wraps=os.posix_fadvise) as advise:
                result = release_consumed_input(view)
            lo = ((8 + 21 * 400 + mmap.PAGESIZE - 1) // mmap.PAGESIZE) * mmap.PAGESIZE
            hi = ((8 + 143 * 400) // mmap.PAGESIZE) * mmap.PAGESIZE
            self.assertEqual(result, {"eligible_bytes": hi-lo, "advised_bytes": hi-lo, "error": None})
            self.assertEqual(advise.call_args.args[1:], (lo, hi-lo, os.POSIX_FADV_DONTNEED))
            np.testing.assert_array_equal(values, data)
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), before)

    def test_large_offset_nested_slice(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data, values = self.fixture(directory, offset=16392)
            view = values[3:300][17:100]
            result = release_consumed_input(view)
            self.assertGreater(result["advised_bytes"], 0)
            self.assertIsNone(result["error"])
            np.testing.assert_array_equal(view, data[20:103])

    def test_skip_heap_strided_empty_and_partial_page(self):
        with tempfile.TemporaryDirectory() as directory:
            _, _, values = self.fixture(directory)
            for view in (np.ones((100, 100)), values[::2], values[:, ::2],
                         values[:0], values[1:2], values.copy()):
                with self.subTest(shape=view.shape), patch("os.posix_fadvise") as advise:
                    self.assertEqual(release_consumed_input(view)["advised_bytes"], 0)
                    advise.assert_not_called()

    def test_skip_writable_and_copy_on_write_maps(self):
        for mode in ("r+", "c"):
            with self.subTest(mode=mode), tempfile.TemporaryDirectory() as directory:
                _, _, values = self.fixture(directory, mode=mode)
                values[0, 0] = 99
                with patch("os.posix_fadvise") as advise:
                    self.assertEqual(release_consumed_input(values)["advised_bytes"], 0)
                    advise.assert_not_called()
                self.assertEqual(values[0, 0], 99)

    def test_advisory_error_is_recorded_without_data_loss(self):
        with tempfile.TemporaryDirectory() as directory:
            _, data, values = self.fixture(directory)
            with patch("os.posix_fadvise", side_effect=OSError("unsupported")):
                receipt = release_consumed_input(values)
            self.assertEqual(receipt["advised_bytes"], 0)
            self.assertGreater(receipt["eligible_bytes"], 0)
            self.assertIn("unsupported", receipt["error"])
            np.testing.assert_array_equal(values, data)

    def test_missing_path_is_best_effort(self):
        with tempfile.TemporaryDirectory() as directory:
            path, data, values = self.fixture(directory)
            path.unlink()
            receipt = release_consumed_input(values)
            self.assertEqual(receipt["advised_bytes"], 0)
            self.assertIsNotNone(receipt["error"])
            np.testing.assert_array_equal(values, data)


if __name__ == "__main__":
    unittest.main()
