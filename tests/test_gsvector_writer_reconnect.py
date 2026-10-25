import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
import psycopg
from psycopg import sql
from psycopg.pq import TransactionStatus

from neurips23.streaming.gsvector.module import Gsvector


class WriterReconnectTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"GSVECTOR_PG_DSN": "dbname=not_connected",
                                          "GSVECTOR_ARTIFACTS": self.tmp.name})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def adapter(self, enabled=True):
        adapter = Gsvector("euclidean", {"reconnect_writer": enabled,
            "build_threads": 8, "compact_max_rows": 500000, "build_neighbor_codes": True})
        adapter.conn = MagicMock()
        adapter.conn.info.transaction_status = TransactionStatus.IDLE
        adapter.schema = "fixture"
        self.addCleanup(adapter.done)
        return adapter

    def test_default_off_and_argument_validation(self):
        adapter = self.adapter(False)
        with patch("neurips23.streaming.gsvector.module.psycopg.connect") as connect:
            adapter._reconnect_writer()
            connect.assert_not_called()
        adapter.conn.close.assert_not_called()
        for value in (1, "on", None):
            with self.assertRaises(ValueError):
                Gsvector("euclidean", {"reconnect_writer": value})

    def test_live_transaction_is_not_closed(self):
        adapter = self.adapter()
        adapter.conn.info.transaction_status = TransactionStatus.INTRANS
        with self.assertRaisesRegex(RuntimeError, "committed"):
            adapter._reconnect_writer()
        adapter.conn.close.assert_not_called()

    def test_configuration_failure_preserves_original_connection(self):
        adapter = self.adapter()
        old, new = adapter.conn, MagicMock()
        with patch("neurips23.streaming.gsvector.module.psycopg.connect", return_value=new), \
                patch.object(adapter, "_configure_write_connection", side_effect=RuntimeError("configuration")):
            with self.assertRaisesRegex(RuntimeError, "configuration"):
                adapter._reconnect_writer()
        self.assertIs(adapter.conn, old)
        old.close.assert_not_called()
        new.close.assert_called_once()

    def test_new_writer_reapplies_settings_without_closing_query_pool(self):
        adapter = self.adapter()
        old, new = adapter.conn, MagicMock()
        query_connection = MagicMock()
        adapter._query_connections = [query_connection]
        adapter._single_state = ("old",)
        with patch("neurips23.streaming.gsvector.module.psycopg.connect", return_value=new):
            adapter._reconnect_writer()
        self.assertIs(adapter.conn, new)
        old.close.assert_called_once()
        query_connection.close.assert_not_called()
        self.assertFalse(hasattr(adapter, "_single_state"))
        self.assertEqual(adapter.maintenance_totals["writer_reconnect_calls"], 1)
        calls = new.execute.call_args_list
        self.assertTrue(any(c.args[0] == "SELECT set_config(%s,%s,false)" and
                            c.args[1] == ("gsvector.build_threads", "8") for c in calls))
        self.assertTrue(any(isinstance(c.args[0], str) and
                            "CREATE TEMP TABLE" in c.args[0] for c in calls))


@unittest.skipUnless(os.environ.get("GSVECTOR_PG_DSN"), "requires an owned test database")
class WriterReconnectIntegrationTests(unittest.TestCase):
    def test_mutations_old_snapshots_and_query_pool_survive_reconnect(self):
        vectors = np.random.default_rng(104).normal(size=(256, 32)).astype("float32")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "GSVECTOR_ARTIFACTS": directory, "GSVECTOR_KEEP_SCHEMA": "0"}):
            adapter = Gsvector("euclidean", {
                "R": 8, "L": 32, "build_threads": 2, "batch_rows": 64, "min_run_rows": 0,
                "vacuum_memtables": True, "reconnect_writer": True, "compact_max_rows": 256,
            })
            try:
                adapter.setup("float32", 256, 32)
                initial_pid = adapter.conn.info.backend_pid
                adapter.set_query_arguments({"search_l": 100, "threads": 2, "reuse_connections": True})
                adapter.insert(vectors[:128], np.arange(128))
                self.assertNotEqual(adapter.conn.info.backend_pid, initial_pid)
                adapter.insert(vectors[128:], np.arange(128, 256))
                adapter.query(vectors[64:80], 10)
                reference = adapter.res.copy()
                query_pids = {c.info.backend_pid for c in adapter._query_connections}
                adapter._reconnect_writer()
                adapter.query(vectors[64:80], 10)
                np.testing.assert_array_equal(adapter.res, reference)
                self.assertEqual({c.info.backend_pid for c in adapter._query_connections}, query_pids)
                with psycopg.connect(adapter.dsn, autocommit=True) as old:
                    old.execute(sql.SQL("SET search_path TO {},public").format(sql.Identifier(adapter.schema)))
                    old.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
                    self.assertEqual(old.execute("SELECT count(*) FROM base").fetchone()[0], 256)
                    adapter.delete(np.arange(32))
                    self.assertEqual(old.execute("SELECT count(*) FROM base").fetchone()[0], 256)
                    old.execute("ROLLBACK")
                # A recreated incoming temp table must still support UPDATE.
                adapter.replace(vectors[64:68], np.arange(64, 68))
                adapter.query(vectors[64:80], 10)
                self.assertTrue((adapter.res >= 32).all())
                self.assertGreaterEqual(adapter.maintenance_totals["writer_reconnect_calls"], 5)
            finally:
                adapter.done()


if __name__ == "__main__":
    unittest.main()
