import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import psycopg
from psycopg import sql

from neurips23.streaming.gsvector.module import Gsvector


class AllocatorWiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"GSVECTOR_PG_DSN": "dbname=not_connected",
                                          "GSVECTOR_ARTIFACTS": self.tmp.name})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def adapter(self, enabled=False):
        adapter = Gsvector("euclidean", {"trim_allocator": enabled})
        adapter.conn = MagicMock()
        adapter.dim = 32
        self.addCleanup(adapter.done)
        return adapter

    def test_default_off_and_validation(self):
        adapter = self.adapter()
        adapter._trim_allocator()
        adapter.conn.execute.assert_not_called()
        for value in (None, 1, "on"):
            with self.assertRaises(ValueError):
                Gsvector("euclidean", {"trim_allocator": value})

    def test_receipts_are_calls_not_claimed_bytes(self):
        adapter = self.adapter(True)
        adapter.conn.execute.return_value.fetchone.side_effect = [(True,), (False,)]
        adapter._trim_allocator()
        adapter._trim_allocator()
        self.assertEqual(adapter.maintenance_totals["allocator_trim_calls"], 2)
        self.assertEqual(adapter.maintenance_totals["allocator_trim_reported_releases"], 1)
        self.assertGreaterEqual(adapter.maintenance_totals["allocator_trim_s"], 0)
        self.assertEqual(len(adapter.conn.execute.call_args_list), 2)

    def test_index_build_keeps_analyze_and_trims_afterwards(self):
        for enabled in (False, True):
            adapter = self.adapter(enabled)
            adapter.conn.execute.return_value.fetchone.return_value = (True,)
            adapter._build_index()
            calls = adapter.conn.execute.call_args_list
            self.assertEqual(calls[1].args[0], "ANALYZE base")
            self.assertEqual(len(calls), 3 if enabled else 2)
            if enabled:
                self.assertEqual(calls[-1].args[0], "SELECT public.gs_trim_allocator()")

    def test_trim_is_last_maintenance_operation(self):
        adapter = self.adapter(True)
        adapter.schema = "fixture"
        adapter.index_ready = True
        adapter._runs = MagicMock(return_value=[(1, 64, 1)])
        adapter.conn.execute.return_value.fetchone.return_value = (True,)
        adapter._maintenance()
        self.assertEqual(adapter.conn.execute.call_args_list[-1].args[0],
                         "SELECT public.gs_trim_allocator()")
        self.assertEqual(adapter.maintenance_totals["allocator_trim_calls"], 1)

    def test_missing_helper_rejected_before_schema_or_data_creation(self):
        adapter = self.adapter(True)
        connection = MagicMock()
        connection.execute.return_value.fetchone.side_effect = [(1234,), (False,)]
        with patch("neurips23.streaming.gsvector.module.psycopg.connect",
                   return_value=connection), self.assertRaisesRegex(ValueError, "allocator helper"):
            adapter.setup("float32", 64, 32)
        self.assertTrue(connection.close.called)
        self.assertEqual(adapter.schemas, [])


@unittest.skipUnless(os.environ.get("GSVECTOR_PG_DSN"), "requires an owned test database")
class AllocatorIntegrationTests(unittest.TestCase):
    def test_queries_and_old_snapshot_survive_allocator_release(self):
        with psycopg.connect(os.environ["GSVECTOR_PG_DSN"], autocommit=True) as c:
            c.execute("CREATE OR REPLACE FUNCTION public.gs_trim_allocator() RETURNS boolean "
                      "AS '$libdir/gsvector','gsvector_trim_allocator' LANGUAGE C VOLATILE")
        vectors = np.random.default_rng(103).normal(size=(256, 32)).astype("float32")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "GSVECTOR_ARTIFACTS": directory, "GSVECTOR_KEEP_SCHEMA": "0"}):
            adapter = Gsvector("euclidean", {
                "R": 8, "L": 32, "build_threads": 4, "batch_rows": 64,
                "min_run_rows": 0, "vacuum_memtables": True, "trim_allocator": True,
            })
            try:
                adapter.setup("float32", 256, 32)
                adapter.insert(vectors, np.arange(256))
                adapter._set_query_arguments({"search_l": 100, "threads": 2,
                                              "reuse_connections": True},
                                             reset_workload=False)
                adapter.query(vectors[:64], 10)
                reference = adapter.res.copy()
                adapter._trim_allocator()
                adapter.query(vectors[:64], 10)
                np.testing.assert_array_equal(adapter.res, reference)
                with psycopg.connect(adapter.dsn, autocommit=True) as old:
                    old.execute(sql.SQL("SET search_path TO {},public").format(
                        sql.Identifier(adapter.schema)))
                    old.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
                    self.assertEqual(old.execute("SELECT count(*) FROM base").fetchone()[0], 256)
                    adapter.delete(np.arange(32))
                    self.assertEqual(old.execute("SELECT count(*) FROM base").fetchone()[0], 256)
                    adapter.query(vectors[32:96], 10)
                    self.assertTrue((adapter.res >= 32).all())
                    reference = adapter.res.copy()
                    adapter._trim_allocator()
                    adapter.query(vectors[32:96], 10)
                    np.testing.assert_array_equal(adapter.res, reference)
                    old.execute("ROLLBACK")
                self.assertGreaterEqual(adapter.maintenance_totals["allocator_trim_calls"], 4)
            finally:
                adapter.done()


if __name__ == "__main__":
    unittest.main()
