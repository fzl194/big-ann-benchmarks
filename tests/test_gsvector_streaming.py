import json
import os
import struct
import tempfile
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import psycopg

from neurips23.streaming.gsvector.module import (
    Gsvector, binary_copy_blocks, encode_vector, has_index_plan,
)


class EncodingTests(unittest.TestCase):
    def test_pq_subspace_validation_and_default_sql(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "GSVECTOR_PG_DSN": "dbname=not_connected", "GSVECTOR_ARTIFACTS": directory,
        }):
            for value in (-1, 65529, True, 1.5, "25"):
                with self.assertRaisesRegex(ValueError, "pq_m"):
                    Gsvector("euclidean", {"pq_m": value})
            for m in (0, 25, 3):
                adapter = Gsvector("euclidean", {"pq_m": m})
                adapter.dim = 100
                adapter.conn = MagicMock()
                try:
                    if m == 3:
                        with self.assertRaisesRegex(ValueError, "divide"):
                            adapter._build_index()
                        adapter.conn.execute.assert_not_called()
                    else:
                        adapter._build_index()
                        text = adapter.conn.execute.call_args_list[0].args[0].as_string()
                        if m:
                            self.assertIn("pq_m=25", text)
                        else:
                            self.assertNotIn("pq_m", text)
                finally:
                    adapter.done()

    def test_bounded_maintenance_and_pool_parameter_validation(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "GSVECTOR_PG_DSN": "dbname=not_connected",
            "GSVECTOR_ARTIFACTS": directory,
        }):
            for value in (-1, 2147483648):
                with self.assertRaises(ValueError):
                    Gsvector("euclidean", {"compact_max_rows": value})
            for name in ("build_beam_first", "build_neighbor_codes"):
                with self.assertRaises(ValueError):
                    Gsvector("euclidean", {name: "on"})
            adapter = Gsvector("euclidean", {"compact_max_rows": 500000})
            try:
                self.assertEqual(adapter.compact_max_rows, 500000)
                for value in (-1, 1025):
                    with self.assertRaises(ValueError):
                        adapter.set_query_arguments({"fed_run_pool_max": value})
                adapter.set_query_arguments({"fed_run_pool_max": 64})
                self.assertEqual(adapter.query_args["fed_run_pool_max"], 64)
            finally:
                adapter.done()

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
    def test_fraction_reclamation_threshold_and_absolute_floor(self):
        for rows, fraction, absolute, threshold in ((64, .2, 1, 13),
                                                     (64, .2, 20, 20),
                                                     (100, .07, 1, 7)):
            with self.subTest(rows=rows, fraction=fraction, absolute=absolute), \
                 tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                     "GSVECTOR_ARTIFACTS": directory, "GSVECTOR_KEEP_SCHEMA": "0"}):
                vectors = np.random.default_rng(20261017).normal(size=(rows, 32)).astype("float32")
                adapter = Gsvector("euclidean", {
                    "R": 8, "L": 32, "min_run_rows": 0, "batch_rows": rows,
                    "build_threads": 1, "compact_runs": 4, "compaction": "tiered",
                    "compact_max_rows": rows, "reclaim_max_rows": rows, "reclaim_cooldown": 3,
                    "reclaim_min_dead": absolute, "reclaim_min_dead_fraction": fraction,
                })
                try:
                    adapter.setup("float32", rows * 2, 32)
                    adapter.insert(vectors, np.arange(rows, dtype=np.uint32))
                    original = adapter._runs()
                    # One row below the effective threshold: scan, but no rebuild.
                    adapter.delete(np.arange(1, threshold, dtype=np.uint32))
                    self.assertEqual(adapter._runs(), original)
                    event = [e for e in adapter.events if e["operation"] == "maintenance_reclaim"][-1]
                    self.assertEqual(event["outcomes"][0]["min_reclaimable_rows"], threshold)
                    self.assertEqual(event["outcomes"][0]["outcome"], "insufficient_reclaimable")
                    self.assertEqual(adapter.maintenance_totals["reclaim_calls"], 1)
                    # Exactly the threshold: wait out the unchanged cooldown.
                    adapter.delete(np.array([threshold], dtype=np.uint32))
                    self.assertEqual(adapter.maintenance_totals["reclaim_calls"], 1)
                    adapter._maintenance()
                    self.assertEqual(adapter.maintenance_totals["reclaim_calls"], 1)
                    adapter._maintenance()
                    self.assertEqual(adapter.maintenance_totals["reclaim_calls"], 2)
                    self.assertEqual(sum(r[1] for r in adapter._runs()), rows - threshold)
                    event = [e for e in adapter.events if e["operation"] == "maintenance_reclaim"][-1]
                    self.assertEqual(event["outcomes"][0]["outcome"], "reclaimed")
                    self.assertEqual(event["outcomes"][0]["min_reclaimable_rows"], threshold)
                    active = np.r_[0, np.arange(threshold + 1, rows)]
                    adapter.query(vectors[active[:8]], 8)
                    result = adapter.get_results()
                    np.testing.assert_array_equal(result[:, 0], active[:8])
                    self.assertTrue(np.isin(result, active).all())
                finally:
                    adapter.done()

    def test_opt_in_single_run_reclamation_and_default_compatibility(self):
        vectors = np.random.default_rng(20261017).normal(size=(64, 32)).astype("float32")
        for budget in (0, 63, 64):
            with self.subTest(budget=budget), tempfile.TemporaryDirectory() as directory, \
                 patch.dict(os.environ, {"GSVECTOR_ARTIFACTS": directory, "GSVECTOR_KEEP_SCHEMA": "0"}):
                adapter = Gsvector("euclidean", {
                    "R": 8, "L": 32, "min_run_rows": 0, "batch_rows": 64,
                    "build_threads": 1, "compact_runs": 4, "compaction": "tiered",
                    "compact_max_rows": 64, "reclaim_max_rows": budget,
                    "reclaim_cooldown": 3,
                })
                try:
                    adapter.setup("float32", 128, 32)
                    adapter.insert(vectors, np.arange(64, dtype=np.uint32))
                    self.assertNotIn("reclaim_calls", adapter.maintenance_totals)
                    adapter.delete(np.arange(8, 64, dtype=np.uint32))
                    self.assertEqual(sum(r[1] for r in adapter._runs()), 8 if budget == 64 else 64)
                    adapter.query(vectors[:8], 8)
                    np.testing.assert_array_equal(adapter.get_results()[:, 0], np.arange(8))
                    self.assertTrue(np.all(adapter.get_results() < 8))
                    if budget == 64:
                        self.assertEqual(adapter.maintenance_totals["reclaim_calls"], 1)
                        for _ in range(2):
                            adapter._maintenance()
                        self.assertEqual(adapter.maintenance_totals["reclaim_calls"], 1)
                        adapter._maintenance()
                        self.assertEqual(adapter.maintenance_totals["reclaim_calls"], 2)
                        self.assertEqual(adapter.events[-1]["outcomes"][0]["outcome"],
                                         "insufficient_reclaimable")
                        adapter.set_query_arguments({"search_l": 100, "threads": 1})
                        self.assertFalse(adapter._reclaim_dirty)
                        self.assertEqual(adapter.reclaim_planner.tick, 0)
                        self.assertEqual(adapter.reclaim_planner.due, {})
                    elif budget:
                        event = [e for e in adapter.events if e["operation"] == "maintenance_reclaim"][-1]
                        self.assertEqual(event["planned_input_rows"], 0)
                        self.assertEqual(len(event["oversized_runs"]), 1)
                        self.assertNotIn("reclaim_calls", adapter.maintenance_totals)
                    else:
                        self.assertFalse(any(e["operation"] == "maintenance_reclaim" for e in adapter.events))
                finally:
                    adapter.done()

    def test_reclamation_retries_after_old_snapshot_release_without_new_delete(self):
        vectors = np.random.default_rng(20261018).normal(size=(64, 32)).astype("float32")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "GSVECTOR_ARTIFACTS": directory, "GSVECTOR_KEEP_SCHEMA": "0"}):
            adapter = Gsvector("euclidean", {
                "R": 8, "L": 32, "min_run_rows": 0, "batch_rows": 64,
                "build_threads": 1, "compact_runs": 4, "compaction": "tiered",
                "compact_max_rows": 64, "reclaim_max_rows": 64, "reclaim_cooldown": 3,
            })
            try:
                adapter.setup("float32", 128, 32)
                adapter.insert(vectors, np.arange(64, dtype=np.uint32))
                with psycopg.connect(adapter.dsn, autocommit=True) as old:
                    old.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
                    self.assertEqual(old.execute(
                        f'SELECT count(*) FROM "{adapter.schema}".base').fetchone()[0], 64)
                    adapter.delete(np.arange(8, 64, dtype=np.uint32))
                    self.assertEqual(sum(r[1] for r in adapter._runs()), 64)
                    event = [e for e in adapter.events if e["operation"] == "maintenance_reclaim"][-1]
                    self.assertEqual(event["outcomes"][0]["outcome"], "insufficient_reclaimable")
                    old.execute("ROLLBACK")
                for _ in range(3):
                    adapter._maintenance()
                self.assertEqual(sum(r[1] for r in adapter._runs()), 8)
                adapter.query(vectors[:8], 8)
                np.testing.assert_array_equal(adapter.get_results()[:, 0], np.arange(8))
            finally:
                adapter.done()

    def test_explicit_pq_subspaces_survive_mutation_and_attach(self):
        from benchmark.algorithms.pg_dense import PGDenseANN
        vectors = np.random.default_rng(125).normal(size=(128, 100)).astype("float32")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "GSVECTOR_ARTIFACTS": directory, "GSVECTOR_KEEP_SCHEMA": "0",
        }):
            adapter = Gsvector("euclidean", {"R": 16, "L": 64, "pq_m": 25,
                "min_run_rows": 0, "build_threads": 2, "batch_rows": 32, "compact_runs": 2})
            try:
                adapter.setup("float32", 128, 100)
                adapter.insert(vectors[:96], np.arange(96, dtype=np.uint32))
                adapter.insert(vectors[96:], np.arange(96, 128, dtype=np.uint32))
                options = adapter.conn.execute(
                    "SELECT reloptions FROM pg_class WHERE oid=%s::regclass",
                    (adapter.schema+".ann_idx",)).fetchone()[0]
                self.assertIn("pq_m=25", options)
                adapter.query(vectors[[1, 100]], 10)
                np.testing.assert_array_equal(adapter.get_results()[:, 0], [1, 100])
                reused = PGDenseANN("euclidean", {})
                try:
                    reused.attach(adapter.schema)
                    self.assertEqual(reused.pq_m, 25)
                    reused.set_query_arguments({"search_l": 200, "threads": 1})
                    np.testing.assert_array_equal(reused.query(vectors[[1, 100]], 10)[:, 0],
                                                  [1, 100])
                finally:
                    reused.done()
            finally:
                adapter.done()

    def test_reused_connections_follow_mutations_and_close_on_reset(self):
        rng = np.random.default_rng(20261015)
        vectors = rng.normal(size=(160, 16)).astype("float32")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "GSVECTOR_ARTIFACTS": directory, "GSVECTOR_KEEP_SCHEMA": "0",
            "GSVECTOR_SAVE_CHECKPOINTS": "1",
        }):
            adapter = Gsvector("euclidean", {
                "R": 16, "L": 64, "min_run_rows": 0, "batch_rows": 32,
                "build_threads": 2, "compact_runs": 4,
            })
            try:
                adapter.setup("float32", 160, 16)
                adapter.set_query_arguments({"search_l": 200, "threads": 2,
                                             "reuse_connections": True})
                adapter.insert(vectors[:96], np.arange(96, dtype=np.uint32))
                adapter.query(vectors[:8], 10)
                np.testing.assert_array_equal(adapter.get_results()[:, 0], np.arange(8))
                saved = adapter.events[-1]["checkpoint_file"]
                np.testing.assert_array_equal(np.load(saved, allow_pickle=False), adapter.get_results())
                connections = list(adapter._query_connections)
                self.assertTrue(connections)
                adapter.query(vectors[:8], 10)
                self.assertEqual(adapter._query_connections, connections)
                adapter.delete(np.arange(16, dtype=np.uint32))
                adapter.replace(vectors[96:112], np.arange(16, 32, dtype=np.uint32))
                adapter.query(vectors[96:104], 10)
                np.testing.assert_array_equal(adapter.get_results()[:, 0], np.arange(16, 24))
                self.assertFalse(np.any(adapter.get_results() < 16))
                adapter.insert(vectors[:16], np.arange(16, dtype=np.uint32))
                vacuum_calls = adapter.maintenance_totals["vacuum_calls"]
                adapter.insert(vectors[112:144], np.arange(112, 144, dtype=np.uint32))
                self.assertGreater(adapter.maintenance_totals["vacuum_calls"], vacuum_calls)
                adapter.query(vectors[:8], 10)
                np.testing.assert_array_equal(adapter.get_results()[:, 0], np.arange(8))
                # Change session settings without replaying the workload.
                adapter._set_query_arguments({"search_l": 100, "threads": 2,
                    "reuse_connections": True, "random_io": True, "fed_run_pool_max": 32}, reset_workload=False)
                adapter.query(vectors[:8], 10)
                for connection in connections:
                    self.assertEqual(connection.execute(
                        "SHOW gsvector.fed_run_pool_max").fetchone()[0], "32")
                adapter._set_query_arguments({"search_l": 100, "threads": 2,
                    "reuse_connections": True}, reset_workload=False)
                adapter.query(vectors[:8], 10)
                for connection in connections:
                    self.assertEqual(connection.execute(
                        "SHOW gsvector.query_random_io").fetchone()[0], "off")
                    self.assertEqual(connection.execute(
                        "SHOW gsvector.fed_run_pool_max").fetchone()[0], "8")
                adapter.set_query_arguments({"search_l": 100, "threads": 1})
                self.assertTrue(all(connection.closed for connection in connections))
                self.assertEqual(adapter.live_count, 0)
                adapter.insert(vectors[:32], np.arange(32, dtype=np.uint32))
                adapter.query(vectors[:4], 10)
                np.testing.assert_array_equal(adapter.get_results()[:, 0], np.arange(4))
            finally:
                adapter.done()
            self.assertFalse(adapter._query_connections)

    def test_threshold_seal_then_explicit_maintenance(self):
        rng=np.random.default_rng(189)
        vectors=rng.normal(size=(2144,16)).astype("float32")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
            "GSVECTOR_ARTIFACTS":directory,"GSVECTOR_KEEP_SCHEMA":"0",
        }):
            adapter=Gsvector("euclidean",{"R":16,"L":32,"min_run_rows":0,
                "batch_rows":1024,"build_threads":2,"compact_runs":4})
            try:
                adapter.setup("float32",len(vectors),16)
                adapter.insert(vectors[:96],np.arange(96,dtype=np.uint32))
                adapter.query(vectors[:2],10)
                # These full-size batches trigger automatic sealing during
                # VACUUM. The following explicit maintenance must drain first.
                for lo in (96,1120):
                    adapter.insert(vectors[lo:lo+1024],np.arange(lo,lo+1024,dtype=np.uint32))
                    adapter.query(vectors[lo:lo+2],10)
                    np.testing.assert_array_equal(adapter.get_results()[:,0],np.arange(lo,lo+2))
                self.assertEqual(adapter.live_count,len(vectors))
            finally:
                adapter.done()

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
