import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from neurips23.streaming.gsvector.module import Gsvector
from benchmark.algorithms.pg_dense import PGDenseANN


GUC = "gsvector.pq_rerank_frontier"
SET_SQL = "SELECT set_config('gsvector.pq_rerank_frontier',%s,false)"


class PqFrontierRerankTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.environment = patch.dict(os.environ, {
            "GSVECTOR_PG_DSN": "dbname=not_connected",
            "GSVECTOR_ARTIFACTS": self.directory.name,
        })
        self.environment.start()
        self.adapter = Gsvector("euclidean", {})

    def tearDown(self):
        self.adapter.done()
        self.environment.stop()
        self.directory.cleanup()

    def configure(self, **options):
        self.adapter.set_query_arguments({"search_l": 1600, "threads": 1, **options})

    @staticmethod
    def connection(vartype="bool"):
        connection = MagicMock()
        connection.execute.return_value.fetchone.return_value = (
            None if vartype is None else (vartype,))
        return connection

    @staticmethod
    def writes(connection):
        return [call for call in connection.execute.call_args_list
                if isinstance(call.args[0], str) and
                (call.args[0].startswith("RESET " + GUC) or
                 call.args[0] == SET_SQL)]

    def test_rejects_non_boolean_query_values_before_any_connection(self):
        for value in (0, 1, None, "on", "false", [], {}, np.bool_(True)):
            with self.subTest(value=value), patch(
                    "neurips23.streaming.gsvector.module.psycopg.connect") as connect:
                with self.assertRaisesRegex(ValueError, "pq_rerank_frontier must be boolean"):
                    self.configure(pq_rerank_frontier=value)
                connect.assert_not_called()

    def test_boolean_values_are_query_metadata_without_changing_search_width(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                self.configure(pq_rerank_frontier=enabled)
                self.assertIs(self.adapter.query_args["pq_rerank_frontier"], enabled)
                self.assertEqual(self.adapter.query_args["search_l"], 1600)
                self.assertIn("pq_rerank_frontier", str(self.adapter))

    def test_new_mode_is_query_only_not_a_build_parameter(self):
        with self.assertRaisesRegex(ValueError, "unknown index parameters"):
            Gsvector("euclidean", {"pq_rerank_frontier": True})

    def test_explicit_on_and_off_require_registered_boolean_guc_then_apply(self):
        for enabled in (True, False):
            with self.subTest(enabled=enabled):
                self.configure(pq_rerank_frontier=enabled)
                connection = self.connection()
                self.adapter.conn = connection
                self.adapter._apply_random_io(connection)
                connection.execute.assert_any_call(
                    "SELECT vartype FROM pg_settings WHERE name = %s", (GUC,))
                connection.execute.assert_any_call(SET_SQL, ("on" if enabled else "off",))
                self.assertEqual(len(self.writes(connection)), 1)

    def test_explicit_request_rejects_old_or_wrong_type_server_before_set(self):
        for enabled in (True, False):
            for vartype in (None, "string", "integer"):
                with self.subTest(enabled=enabled, vartype=vartype):
                    self.configure(pq_rerank_frontier=enabled)
                    connection = self.connection(vartype)
                    self.adapter.conn = connection
                    with self.assertRaisesRegex(RuntimeError, "matching.*extension"):
                        self.adapter._apply_random_io(connection)
                    self.assertEqual(self.writes(connection), [])

    def test_omitted_option_on_fresh_connection_does_not_touch_new_guc(self):
        self.configure()
        connection = self.connection(None)
        self.adapter.conn = connection
        self.adapter._apply_random_io(connection)
        self.assertFalse(any(GUC in str(call)
                             for call in connection.execute.call_args_list))

    def test_single_connection_on_then_omitted_sets_off_once(self):
        self.configure(pq_rerank_frontier=True)
        connection = self.connection()
        self.adapter.conn = connection
        self.adapter._apply_random_io(connection)
        self.configure()
        connection.execute.reset_mock()
        self.adapter._apply_random_io(connection)
        self.assertEqual([call.args for call in self.writes(connection)], [(SET_SQL, ("off",))])
        connection.execute.reset_mock()
        self.adapter._apply_random_io(connection)
        self.assertFalse(any(GUC in str(call)
                             for call in connection.execute.call_args_list))

    def test_single_connection_on_then_explicit_off(self):
        connection = self.connection()
        self.adapter.conn = connection
        self.configure(pq_rerank_frontier=True)
        self.adapter._apply_random_io(connection)
        self.configure(pq_rerank_frontier=False)
        connection.execute.reset_mock()
        self.adapter._apply_random_io(connection)
        self.assertEqual([call.args for call in self.writes(connection)], [(SET_SQL, ("off",))])

    def test_every_reused_connection_resets_on_then_omitted(self):
        self.configure(pq_rerank_frontier=True, reuse_connections=True)
        connections = [self.connection(), self.connection()]
        for connection in connections:
            self.adapter._apply_random_io(connection)
            connection.execute.assert_any_call(SET_SQL, ("on",))
        self.configure(reuse_connections=True)
        for connection in connections:
            connection.execute.reset_mock()
            self.adapter._apply_random_io(connection)
            self.assertEqual([call.args for call in self.writes(connection)], [(SET_SQL, ("off",))])

    def test_omitted_reused_connection_supports_old_server_without_reset(self):
        self.configure(reuse_connections=True)
        connection = self.connection(None)
        self.adapter._apply_random_io(connection)
        self.assertEqual(self.writes(connection), [])
        self.assertFalse(any(isinstance(call.args[0], str) and
                             call.args[0].startswith("RESET " + GUC)
                             for call in connection.execute.call_args_list))

    def test_explicit_option_is_applied_on_each_query_configuration(self):
        self.configure(pq_rerank_frontier=True, reuse_connections=True)
        connection = self.connection()
        for _ in range(2):
            connection.execute.reset_mock()
            self.adapter._apply_random_io(connection)
            self.assertEqual([call.args for call in self.writes(connection)], [(SET_SQL, ("on",))])

    def dense_single_adapter(self):
        self.adapter.done()
        self.adapter = PGDenseANN("euclidean", {})
        self.adapter.schema = "owned_fixture"
        self.adapter.dim = 2
        self.adapter.live_count = 1
        self.adapter.vector_oid = 1234
        self.adapter.keep_schema = True
        self.adapter._runs = MagicMock(return_value=[])
        self.adapter._record = MagicMock()
        connection = self.connection()
        cursor = MagicMock()
        cursor.fetchone.return_value = ("bool",)
        cursor.fetchall.return_value = [(42,)]
        plan = MagicMock()
        plan.fetchone.return_value = ([{"Plan": {
            "Node Type": "Index Scan", "Index Name": "ann_idx"}}],)
        connection.execute.side_effect = lambda statement, *args: (
            plan if isinstance(statement, str) and statement.startswith("EXPLAIN ") else cursor)
        self.adapter.conn = connection
        return connection

    def test_dense_single_false_true_omitted_reconfigures_server_on_same_index(self):
        connection = self.dense_single_adapter()
        query = np.asarray([1, 2], dtype=np.float32)
        for options in ({"pq_rerank_frontier": False},
                        {"pq_rerank_frontier": True}, {}):
            self.configure(**options)
            np.testing.assert_array_equal(self.adapter.query(query, 1), [42])
        self.assertEqual([call.args for call in self.writes(connection)], [
            (SET_SQL, ("off",)), (SET_SQL, ("on",)), (SET_SQL, ("off",))])
        records = self.adapter._record.call_args_list
        self.assertIs(records[0].kwargs["query_args"]["pq_rerank_frontier"], False)
        self.assertIs(records[1].kwargs["query_args"]["pq_rerank_frontier"], True)
        self.assertNotIn("pq_rerank_frontier", records[2].kwargs["query_args"])

    def test_dense_single_true_false_reconfigures_server_without_other_changes(self):
        connection = self.dense_single_adapter()
        query = np.asarray([1, 2], dtype=np.float32)
        for enabled in (True, False):
            self.configure(pq_rerank_frontier=enabled)
            np.testing.assert_array_equal(self.adapter.query(query, 1), [42])
        self.assertEqual([call.args for call in self.writes(connection)], [
            (SET_SQL, ("on",)), (SET_SQL, ("off",))])


if __name__ == "__main__":
    unittest.main()
