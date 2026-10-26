import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock

import h5py
import numpy as np

from neurips23.streaming.gsvector.dynamic import GsvectorDynamic


class DynamicMetadataTests(unittest.TestCase):
    def test_upstream_hdf5_attributes_roundtrip(self):
        adapter = GsvectorDynamic("euclidean", {"query_mode": "index"})
        details = {"schema": "gs_stream_fixture", "query_mode": "index",
                   "revision": 640, "table_bytes": 6498328576,
                   "streaming_index_created": True,
                   "scope": "actual ORDER BY index-AM scans + dynamic worker"}
        adapter.backend = Mock()
        adapter.backend.get_additional.return_value = {"pg_dynamic_diskann": details}
        attrs = adapter.get_additional()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "upstream.hdf5"
            with h5py.File(path, "w") as output:
                # Exact upstream store_results attribute assignment contract.
                for key, value in attrs.items():
                    output.attrs[key] = value
                output["neighbors_step2"] = np.array([[1, 2, 3]], dtype=np.uint32)
            with h5py.File(path) as result:
                self.assertEqual(json.loads(result.attrs["pg_dynamic_diskann"]), details)
                self.assertEqual(result.attrs["pg_schema"], details["schema"])
                self.assertEqual(result.attrs["query_mode"], "index")
                np.testing.assert_array_equal(result["neighbors_step2"], [[1, 2, 3]])
        # Serialization must not mutate the backend's structure.
        self.assertIsInstance(details["revision"], int)

    def test_deterministic_json(self):
        adapter = GsvectorDynamic("euclidean", {})
        adapter.backend = Mock()
        adapter.backend.get_additional.return_value = {
            "pg_dynamic_diskann": {"schema": "gs_stream_fixture", "query_mode": "batch"}}
        self.assertEqual(adapter.get_additional(), adapter.get_additional())


if __name__ == "__main__":
    unittest.main()
