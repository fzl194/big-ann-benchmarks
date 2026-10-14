import os
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from scipy.sparse import csr_matrix

from benchmark.algorithms.pg_exact import PGFilterExact, PGSparseExact
from benchmark.algorithms.pg_dense import PGDenseANN
from benchmark.dataset_io import write_sparse_matrix
from benchmark.datasets import DATASETS
from benchmark.local_dataset import LocalDenseDataset, LocalCSRDataset
from neurips23.ood.gsvector.module import GsvectorOOD


def write_bin(path, array):
    with path.open("wb") as f:
        np.asarray(array.shape, dtype="<u4").tofile(f)
        array.astype("<f4").tofile(f)


class SparseValidationTests(unittest.TestCase):
    def test_negative_and_nonfinite_rejected(self):
        for value in (-1, float("nan"), float("inf")):
            with self.assertRaises(ValueError):
                PGSparseExact.validate_csr(csr_matrix([[value, 0]]))

    def test_sparse_shape_preserved(self):
        matrix = csr_matrix(([1.0, 2.0], ([0, 1], [1, 30008])), shape=(2, 30109))
        result = PGSparseExact.validate_csr(matrix)
        self.assertEqual(result.shape, (2, 30109))
        self.assertEqual(result.nnz, 2)


@unittest.skipUnless(os.environ.get("GSVECTOR_PG_DSN"), "requires private PostgreSQL")
class StaticIntegrationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.environment = patch.dict(os.environ, {
            "GSVECTOR_ARTIFACTS": str(self.root/"events"), "GSVECTOR_KEEP_SCHEMA": "0",
        })
        self.environment.start()
        self.names = []

    def tearDown(self):
        for name in self.names:
            DATASETS.pop(name)
        self.environment.stop()
        self.tmp.cleanup()

    def register(self, name, spec, sparse=False):
        self.names.append(name)
        cls = LocalCSRDataset if sparse else LocalDenseDataset
        spec["groundtruth_dir"] = str(self.root/"gt")
        DATASETS[name] = lambda: cls(name, spec)
        return DATASETS[name]()

    def test_array_cosine_sweeps_and_remote_storage_accounting(self):
        rng = np.random.default_rng(17)
        base = rng.normal(size=(96, 31)).astype("float32")
        adapter = PGDenseANN("angular", {"R":16, "L":64, "build_threads":2,
                                         "min_run_rows":0})
        try:
            adapter.fit(base)
            schema = adapter.schema
            snapshots = []
            backend = adapter.conn.execute("SELECT pg_backend_pid()").fetchone()[0]
            for search_l in (100, 200):
                adapter.set_query_arguments(search_l, 2)
                with patch("psycopg.connect",side_effect=AssertionError("single query opened a new backend")):
                    found = adapter.query(base[7], 10)
                self.assertEqual(found.shape, (10,))
                self.assertEqual(found[0], 7)
                self.assertEqual(adapter.conn.execute("SELECT pg_backend_pid()").fetchone()[0],backend)
                adapter.batch_query(base[:3], 10)
                np.testing.assert_array_equal(adapter.get_batch_results()[:,0],np.arange(3))
                self.assertEqual(adapter.schema, schema)
                snapshots.append(adapter.get_additional())
            for snapshot in snapshots:
                self.assertEqual(hashlib.sha256(Path(snapshot["pg_events_file"]).read_bytes()).hexdigest(),
                                 snapshot["pg_events_sha256"])
            storage = adapter.get_storage_usage()
            self.assertGreater(storage["external_index_bytes"], 0)
            self.assertGreater(storage["internal_relations_bytes"], 0)
            self.assertEqual(storage["index_total_bytes"],
                             storage["external_index_bytes"]+storage["internal_relations_bytes"]
                             +storage["pg_index_bytes"])
            attached = PGDenseANN("cosine", {})
            try:
                attached.attach(schema)
                self.assertEqual(attached.query(base[7],10)[0],7)
                self.assertEqual(attached.live_count,len(base))
            finally:
                attached.done()
            self.assertEqual(adapter.conn.execute("SELECT count(*) FROM base").fetchone()[0],len(base))
            mismatched = PGDenseANN("ip", {})
            try:
                with self.assertRaises(ValueError):
                    mismatched.attach(schema)
            finally:
                mismatched.done()
        finally:
            adapter.done()

    def test_native_ip_and_query_arg_sweep(self):
        base = np.zeros((96, 200), dtype="float32")
        base[:, 0] = np.arange(1, 97)
        queries = np.zeros((2, 200), dtype="float32")
        queries[:, 0] = [1, -1]
        write_bin(self.root/"base.bin", base)
        write_bin(self.root/"queries.bin", queries)
        name = "_test_pg_native_ip"
        self.register(name, {"base": str(self.root/"base.bin"), "queries": str(self.root/"queries.bin"),
                             "metric": "ip"})
        adapter = GsvectorOOD("ip", {"R":16, "L":64, "min_run_rows":0, "build_threads":2})
        try:
            adapter.fit(name)
            schema = adapter.schema
            for threads in (1, 2):
                adapter.set_query_arguments({"search_l":200, "threads":threads})
                adapter.query(queries, 10)
                np.testing.assert_array_equal(adapter.get_results()[:, 0], [95, 0])
                self.assertEqual(adapter.schema, schema)
                self.assertEqual(adapter.conn.execute("SELECT count(*) FROM base").fetchone()[0], 96)
        finally:
            adapter.done()

    def test_label_and_empty_and_missing_filters(self):
        base = np.repeat(np.arange(8, dtype="float32")[:, None], 192, axis=1)
        queries = np.zeros((4, 192), dtype="float32")
        write_bin(self.root/"base.bin", base)
        write_bin(self.root/"queries.bin", queries)
        labels = [[1,2], [1], [2], [1,2], [3], [], [1], [2]]
        query_labels = [[1], [1,2], [0], []]
        for name, rows in [("base.meta", labels), ("query.meta", query_labels)]:
            rr, cc = [], []
            for i, terms in enumerate(rows):
                rr.extend([i]*len(terms)); cc.extend(terms)
            write_sparse_matrix(csr_matrix((np.ones(len(rr)), (rr,cc)), shape=(len(rows),4)),
                                str(self.root/name))
        name = "_test_pg_filter"
        ds = self.register(name, {"base": str(self.root/"base.bin"), "queries": str(self.root/"queries.bin"),
                                 "base_metadata": str(self.root/"base.meta"),
                                 "query_metadata": str(self.root/"query.meta")})
        adapter = PGFilterExact("euclidean", {})
        try:
            adapter.fit(name)
            adapter.filtered_query(queries, ds.get_queries_metadata(), 3)
            np.testing.assert_array_equal(adapter.get_results(),
                                          [[0,1,3], [0,3,-1], [-1,-1,-1], [0,1,2]])
        finally:
            adapter.done()

    def test_sparse_high_terms_and_zero_score_fill(self):
        base = csr_matrix(([2,4,5,1,1], ([0,0,1,2,2], [1,30008,2,1,2])),
                          shape=(4,30109), dtype="float32")
        queries = csr_matrix(([1,1,1,1], ([0,1,1,3], [30008,1,2,3])),
                             shape=(4,30109), dtype="float32")
        write_sparse_matrix(base, str(self.root/"base.csr"))
        write_sparse_matrix(queries, str(self.root/"query.csr"))
        name = "_test_pg_sparse"
        self.register(name, {"base": str(self.root/"base.csr"), "queries": str(self.root/"query.csr")},
                      sparse=True)
        adapter = PGSparseExact("ip", {})
        try:
            adapter.fit(name)
            adapter.query(queries, 3)
            np.testing.assert_array_equal(adapter.get_results(),
                                          [[0,1,2], [1,0,2], [0,1,2], [0,1,2]])
        finally:
            adapter.done()


if __name__ == "__main__":
    unittest.main()
