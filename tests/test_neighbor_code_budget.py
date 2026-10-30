import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from neurips23.streaming.gsvector.module import Gsvector
from benchmark.algorithms.pg_dense import PGArrayANN


class NeighborCodeBudgetTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.environment = patch.dict(os.environ, {
            "GSVECTOR_PG_DSN": "dbname=not_connected",
            "GSVECTOR_ARTIFACTS": self.directory.name,
        })
        self.environment.start()

    def tearDown(self):
        self.environment.stop()
        self.directory.cleanup()

    def test_invalid_budgets_fail_before_connecting(self):
        for value in (-2, 2147483648, True, False, 2.5, "2048", None):
            with self.subTest(value=value), patch(
                    "neurips23.streaming.gsvector.module.psycopg.connect") as connect:
                with self.assertRaisesRegex(ValueError, "build_neighbor_code_bytes"):
                    Gsvector("euclidean", {"build_neighbor_code_bytes": value})
                connect.assert_not_called()

    def test_explicit_budget_is_applied_on_each_writer_configuration(self):
        for budget in (-1, 0, 15, 2048, 2147483647):
            with self.subTest(budget=budget):
                adapter = Gsvector("euclidean", {"build_neighbor_code_bytes": budget})
                try:
                    for _ in range(2):
                        connection = MagicMock()
                        adapter._configure_write_connection(connection, set_search_path=False)
                        connection.execute.assert_any_call(
                            "SELECT set_config('gsvector.build_neighbor_code_bytes',%s,false)",
                            (str(budget),))
                finally:
                    adapter.done()

    def test_omitted_budget_does_not_require_new_server_guc(self):
        adapter = Gsvector("euclidean", {})
        try:
            connection = MagicMock()
            adapter._configure_write_connection(connection, set_search_path=False)
            self.assertFalse(any("build_neighbor_code_bytes" in str(call)
                                 for call in connection.execute.call_args_list))
        finally:
            adapter.done()

    def test_legacy_off_switch_is_forwarded_with_budget(self):
        adapter = Gsvector("euclidean", {
            "build_neighbor_codes": False, "build_neighbor_code_bytes": 2048,
        })
        try:
            connection = MagicMock()
            adapter._configure_write_connection(connection, set_search_path=False)
            connection.execute.assert_any_call(
                "SELECT set_config('gsvector.build_neighbor_codes',%s,false)", ("off",))
            connection.execute.assert_any_call(
                "SELECT set_config('gsvector.build_neighbor_code_bytes',%s,false)", ("2048",))
        finally:
            adapter.done()

    def test_array_runner_can_select_code_precision_and_copy_budget(self):
        adapter = PGArrayANN("normalized", 32, 100, 4, 64, 2048)
        try:
            self.assertEqual(adapter.pq_m, 64)
            self.assertEqual(adapter.build_neighbor_code_bytes, 2048)
            self.assertEqual(adapter.metric, "cosine")
        finally:
            adapter.done()

    def test_array_runner_defaults_preserve_old_parameters(self):
        adapter = PGArrayANN("normalized", 32, 100, 4)
        try:
            self.assertNotIn("build_neighbor_code_bytes", adapter.params)
            self.assertNotIn("pq_m", adapter.params)
        finally:
            adapter.done()

    def test_array_runner_rejects_invalid_precision_types(self):
        for value in (False, True, "64", -1, 65529, 2.0):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "pq_m"):
                PGArrayANN("normalized", 32, 100, 4, value, 2048)

    def test_explicit_budget_rejects_an_old_server_before_build(self):
        adapter = Gsvector("euclidean", {"build_neighbor_code_bytes": 2048})
        try:
            connection = MagicMock()
            connection.execute.return_value.fetchone.return_value = None
            with self.assertRaisesRegex(RuntimeError, "matching.*extension"):
                adapter._configure_write_connection(connection, set_search_path=False)
            self.assertFalse(any("set_config('gsvector.build_neighbor_code_bytes'" in str(call)
                                 for call in connection.execute.call_args_list))
        finally:
            adapter.done()


if __name__ == "__main__":
    unittest.main()
