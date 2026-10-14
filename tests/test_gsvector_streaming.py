import json
import os
import struct
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from neurips23.streaming.gsvector.module import (
    Gsvector, binary_copy_blocks, encode_vector, has_index_plan,
)


class EncodingTests(unittest.TestCase):
    def test_binary_copy_layout(self):
        vectors = np.array([[1.25, -2, 3, 4], [5, 6, 7, 8]], dtype=np.float32)
        ids = np.array([0, 2147483647], dtype=np.int64)
        raw = b"".join(binary_copy_blocks(vectors, ids, batch_size=1))
        self.assertEqual(raw[:11], b"PGCOPY\n\xff\r\n\0")
        offset = 19
        for i in range(2):
            fields, size, tag, vsize, dim, unused = struct.unpack_from("!hiqihh", raw, offset)
            self.assertEqual((fields, size, tag, vsize, dim, unused),
                             (2, 8, int(ids[i]), 20, 4, 0))
            np.testing.assert_array_equal(np.frombuffer(raw, dtype=">f4", count=4,
                                                       offset=offset + 22), vectors[i])
            offset += 38
        self.assertEqual(raw[offset:], struct.pack("!h", -1))
        self.assertEqual(encode_vector(vectors[0]), struct.pack("!hhffff", 4, 0, *vectors[0]))

    def test_invalid_vectors_and_plan(self):
        for vector in ([], [[1, 2]], [float("nan")], [float("inf")]):
            with self.assertRaises(ValueError):
                encode_vector(vector)
        self.assertTrue(has_index_plan([{"Plan": {"Plans": [{"Index Name": "ann_idx"}]}}]))
        self.assertFalse(has_index_plan([{"Plan": {"Node Type": "Seq Scan"}}]))

    def test_metric_and_connection_fail_loudly(self):
        with self.assertRaises(ValueError):
            Gsvector("unsupported-metric", {})
        with patch.dict(os.environ, {}, clear=True), self.assertRaises(ValueError):
            Gsvector("euclidean", {})


@unittest.skipUnless(os.environ.get("GSVECTOR_PG_DSN"), "requires dedicated test PostgreSQL")
class PostgreSQLTests(unittest.TestCase):
    def test_mutations_queries_and_parameter_replay(self):
        rng = np.random.default_rng(42)
        vectors = rng.normal(size=(160, 100)).astype("float32")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "GSVECTOR_ARTIFACTS": directory, "GSVECTOR_KEEP_SCHEMA": "0",
        }):
            adapter = Gsvector("euclidean", {
                "R": 16, "L": 64, "min_run_rows": 0, "batch_rows": 32,
                "build_threads": 2, "compact_runs": 4,
            })
            try:
                adapter.setup("float32", 160, 100)
                # Even when parallel index vacuum is eligible, adapter maintenance
                # must explicitly request serial inline drain.
                adapter.conn.execute("SET min_parallel_index_scan_size=0")
                adapter.set_query_arguments({"search_l": 200, "threads": 2})
                adapter.insert(vectors[:96], np.arange(96, dtype=np.uint32))
                stored = adapter.conn.execute("SELECT embedding::text FROM base WHERE id=7").fetchone()[0]
                np.testing.assert_allclose(np.array(json.loads(stored), dtype="float32"), vectors[7],
                                           rtol=1e-6, atol=1e-7)
                adapter.query(vectors[:8], 10)
                np.testing.assert_array_equal(adapter.get_results()[:, 0], np.arange(8))
                first_schema = adapter.schema
                adapter.delete(np.arange(16, dtype=np.uint32))
                adapter.replace(vectors[96:112], np.arange(16, 32, dtype=np.uint32))
                adapter.query(vectors[96:104], 10)
                np.testing.assert_array_equal(adapter.get_results()[:, 0], np.arange(16, 24))
                self.assertFalse(np.any(adapter.get_results() < 16))
                adapter.insert(vectors[:16], np.arange(16, dtype=np.uint32))
                self.assertEqual(adapter.live_count, 96)
                self.assertEqual(adapter.conn.execute("SELECT count(*) FROM base").fetchone()[0], 96)
                # Duplicate/unknown tags must fail, not silently change the live count.
                with self.assertRaises(ValueError):
                    adapter.delete(np.array([999], dtype=np.uint32))
                self.assertEqual(adapter.live_count, 96)
                adapter.set_query_arguments({"search_l": 100, "threads": 1})
                self.assertNotEqual(adapter.schema, first_schema)
                self.assertEqual(adapter.live_count, 0)
                self.assertEqual(adapter.conn.execute("SELECT count(*) FROM base").fetchone()[0], 0)
                adapter.insert(vectors[:32], np.arange(32, dtype=np.uint32))
                adapter.query(vectors[:4], 10)
                np.testing.assert_array_equal(adapter.get_results()[:, 0], np.arange(4))
            finally:
                adapter.done()


if __name__ == "__main__":
    unittest.main()
