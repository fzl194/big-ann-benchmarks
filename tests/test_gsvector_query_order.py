import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from neurips23.streaming.gsvector.module import Gsvector
from neurips23.streaming.gsvector.query_order import locality_order, query_partitions


class QueryOrderTests(unittest.TestCase):
    def test_permutation_determinism_and_readonly_input(self):
        x = np.random.default_rng(73).normal(size=(103, 7)).astype("float32")
        before = x.copy()
        x.flags.writeable = False
        order = locality_order(x)
        np.testing.assert_array_equal(np.sort(order), np.arange(len(x)))
        np.testing.assert_array_equal(order, locality_order(x))
        np.testing.assert_array_equal(x, before)

    def test_empty_single_duplicate_and_extreme_inputs(self):
        for x in (np.empty((0, 3)), np.ones((1, 3)), np.zeros((39, 3)),
                  np.tile([np.finfo("float64").max, -np.finfo("float64").max], (39, 1))):
            order = locality_order(x)
            np.testing.assert_array_equal(np.sort(order), np.arange(len(x)))

    def test_partitions_cover_each_query_once_and_are_balanced(self):
        x = np.random.default_rng(74).normal(size=(59, 4))
        for threads in (1, 4, 64):
            for ordering in ("input", "locality"):
                for dispatch in ("contiguous", "interleaved"):
                    parts = query_partitions(x, threads, ordering, dispatch)
                    np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(len(x)))
                    self.assertLessEqual(max(map(len, parts)) - min(map(len, parts)), 1)
                    self.assertEqual(query_partitions(x[:0], threads, ordering, dispatch), [])

    def test_invalid_input(self):
        for x in (np.ones(3), np.array([[np.nan]]), np.array([[np.inf]])):
            with self.assertRaises(ValueError):
                locality_order(x)
        for leaf in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                locality_order(np.ones((4, 2)), leaf)
        for args in ((0,), (True,), (2, "bad"), (2, "input", "bad")):
            with self.assertRaises(ValueError):
                query_partitions(np.ones((4, 2)), *args)

    def test_adapter_scatter_preserves_query_identity(self):
        x = np.column_stack((np.arange(67), np.arange(67)[::-1])).astype("float32")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "GSVECTOR_PG_DSN": "dbname=not_connected", "GSVECTOR_ARTIFACTS": directory,
                "GSVECTOR_SAVE_CHECKPOINTS": "0"}):
            for ordering in ("input", "locality"):
                for dispatch in ("contiguous", "interleaved"):
                    adapter = Gsvector("euclidean", {})
                    try:
                        adapter.dim = 2
                        adapter.index_ready = True
                        adapter._runs = MagicMock(return_value=[])
                        adapter._record = MagicMock()
                        adapter._set_query_arguments({
                            "threads": 4, "search_l": 10,
                            "query_order": ordering, "query_dispatch": dispatch,
                        }, reset_workload=False)
                        adapter._query_chunk = lambda values, k: np.column_stack(
                            (values[:, 0], values[:, 0] + 100)).astype("int32")
                        adapter.query(x, 2)
                        np.testing.assert_array_equal(adapter.res, np.column_stack(
                            (np.arange(67), np.arange(67) + 100)))
                        adapter.query(x[:0], 2)
                        self.assertEqual(adapter.res.shape, (0, 2))
                        for bad in ({"query_order": "bad"}, {"query_dispatch": False},
                                    *[{"query_batch_size": x} for x in (0, -1, 33, True, 1.5, "2")]):
                            with self.assertRaises(ValueError):
                                adapter._set_query_arguments(bad, reset_workload=False)
                    finally:
                        adapter.done()


@unittest.skipUnless(os.environ.get("GSVECTOR_PG_DSN"), "requires an owned test database")
class QueryOrderIntegrationTests(unittest.TestCase):
    def test_ordered_ids_match_after_deletion(self):
        rng = np.random.default_rng(83)
        vectors = rng.normal(size=(256, 32)).astype("float32")
        queries = vectors[:67].copy()
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "GSVECTOR_ARTIFACTS": directory, "GSVECTOR_KEEP_SCHEMA": "0"}):
            adapter = Gsvector("euclidean", {
                "R": 8, "L": 32, "build_threads": 1, "batch_rows": 64,
                "compact_runs": 100, "min_run_rows": 0, "vacuum_memtables": True,
            })
            try:
                adapter.setup("float32", 256, 32)
                adapter.insert(vectors, np.arange(256))
                adapter.conn.execute(
                    "CREATE OR REPLACE FUNCTION public.gs_ann_batch(regclass,vector[],int) "
                    "RETURNS TABLE(query_no int,tid tid,dist float8) "
                    "AS '$libdir/gsvector','gsvector_ann_batch' LANGUAGE C STABLE PARALLEL RESTRICTED")
                for deleted in (False, True):
                    if deleted:
                        adapter.delete(np.arange(15))
                    reference = None
                    for ordering, dispatch, batch_size in (
                            ("input", "contiguous", 1), ("locality", "interleaved", 1),
                            ("locality", "contiguous", 1), ("input", "interleaved", 1),
                            ("input", "contiguous", 2), ("input", "contiguous", 8),
                            ("locality", "interleaved", 8), ("input", "contiguous", 1)):
                        adapter._set_query_arguments({
                            "search_l": 100, "threads": 3, "reuse_connections": True,
                            "query_order": ordering, "query_dispatch": dispatch,
                            "query_batch_size": batch_size,
                        }, reset_workload=False)
                        adapter.query(queries, 10)
                        if reference is None:
                            reference = adapter.res.copy()
                        else:
                            np.testing.assert_array_equal(adapter.res, reference)
            finally:
                adapter.done()


if __name__ == "__main__":
    unittest.main()
