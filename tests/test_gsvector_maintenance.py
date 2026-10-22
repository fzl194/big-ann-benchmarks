"""Offline scheduler and adapter wiring checks; no database required."""
import os
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import yaml

from neurips23.streaming.gsvector.maintenance import ReclaimPlanner, ReclaimThreshold
from neurips23.streaming.gsvector.module import Gsvector


class ThresholdTests(unittest.TestCase):
    def test_default_and_absolute_floor(self):
        self.assertEqual(ReclaimThreshold().minimum(2147483647), 1)
        self.assertEqual(ReclaimThreshold(20, .2).minimum(64), 20)
        self.assertEqual(ReclaimThreshold(100, 1).minimum(64), 100)

    def test_decimal_ceiling_and_boundaries(self):
        for fraction, rows, expected in ((.2, 64, 13), (.2, 65, 13), (.2, 66, 14),
                                         (.07, 100, 7), (.07, 101, 8),
                                         (1, 2147483647, 2147483647),
                                         (.2, 2147483647, 429496730),
                                         (1e-20, 2147483647, 1)):
            with self.subTest(fraction=fraction, rows=rows):
                self.assertEqual(ReclaimThreshold(1, fraction).minimum(rows), expected)

    def test_invalid_values(self):
        for value in (-.1, 1.01, True, "0.2", None, float("nan"),
                      float("inf"), -float("inf"), 10**400):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ReclaimThreshold(1, value)
        for value in (0, -1, True, 1.5, 2147483648):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ReclaimThreshold(value)
            with self.subTest(rows=value), self.assertRaises(ValueError):
                ReclaimThreshold().minimum(value)


class PlannerTests(unittest.TestCase):
    def test_backoff_bounded_retry_and_reset(self):
        planner = ReclaimPlanner(64, 1, 2, backoff_max=8)
        attempts = []
        for _ in range(31):
            jobs, _ = planner.plan([(1, 64)])
            if jobs:
                attempts.append(planner.tick)
                planner.record_outcome(1, "insufficient_reclaimable")
        self.assertEqual(attempts, [1, 5, 13, 21, 29])
        self.assertEqual(planner.retry_delays, {1: 8})
        planner.record_outcome(1, "reclaimed")
        self.assertNotIn(1, planner.retry_delays)
        planner.defer(2)
        self.assertEqual(planner.plan([(2, 8)])[0], [])
        self.assertNotIn(1, planner.due)
        planner.reset()
        self.assertEqual(planner.retry_delays, {})
        self.assertEqual(planner.due, {})

    def test_backoff_disabled_preserves_schedule_and_missing_runs_are_pruned(self):
        planner = ReclaimPlanner(64, 1, 2)
        planner.plan([(1, 64)])
        planner.record_outcome(1, "insufficient_reclaimable")
        self.assertEqual(planner.due, {1: 3})
        self.assertEqual(planner.retry_delays, {})
        enabled = ReclaimPlanner(64, 1, 2, 8)
        enabled.plan([(1, 64)])
        enabled.record_outcome(1, "insufficient_reclaimable")
        enabled.plan([(2, 64)])
        self.assertNotIn(1, enabled.retry_delays)
        self.assertNotIn(1, enabled.due)
        with self.assertRaises(ValueError):
            enabled.record_outcome(2, "unknown")

    def test_backoff_validation(self):
        for value in (-1, True, 1.5, "8", 1, 1000001):
            with self.subTest(value=value), self.assertRaises(ValueError):
                ReclaimPlanner(64, 1, 2, value)

    def test_budget_count_and_oversized_reporting(self):
        planner = ReclaimPlanner(64, 2, 3)
        jobs, oversized = planner.plan([("1", 40), ("2", 40), ("3", 24), ("4", 65)])
        self.assertEqual(jobs, [(1, 40), (3, 24)])
        self.assertEqual(oversized, [{"run_id": 4, "input_rows": 65}])
        self.assertEqual(planner.plan([("1", 40), ("2", 40), ("3", 24)])[0], [(2, 40)])
        self.assertNotIn(4, planner.due)

    def test_cooldown_and_new_output_deferral(self):
        planner = ReclaimPlanner(64, 1, 3)
        self.assertEqual(planner.plan([(1, 64)])[0], [(1, 64)])
        self.assertEqual(planner.plan([(1, 64)])[0], [])
        self.assertEqual(planner.plan([(1, 64)])[0], [])
        self.assertEqual(planner.plan([(1, 64)])[0], [(1, 64)])
        planner.defer(2)
        self.assertEqual(planner.plan([(2, 8)])[0], [])
        self.assertNotIn(1, planner.due)
        planner.reset()
        self.assertEqual(planner.due, {})
        self.assertEqual(planner.tick, 0)

    def test_older_due_work_precedes_new_arrivals(self):
        planner = ReclaimPlanner(64, 1, 3)
        runs = [(1, 64), (2, 64), (3, 64)]
        self.assertEqual(planner.plan(runs)[0], [(1, 64)])
        self.assertEqual(planner.plan(runs + [(4, 64)])[0], [(2, 64)])
        self.assertEqual(planner.plan(runs + [(4, 64), (5, 64)])[0], [(3, 64)])
        self.assertEqual(planner.plan(runs + [(4, 64), (5, 64), (6, 64)])[0], [(4, 64)])

    def test_fresh_compaction_output_not_scanned_immediately(self):
        planner = ReclaimPlanner(64, 1, 3)
        self.assertEqual(planner.plan([(10, 64)], {10})[0], [])
        self.assertEqual(planner.plan([(10, 64)])[0], [])
        self.assertEqual(planner.plan([(10, 64)])[0], [])
        self.assertEqual(planner.plan([(10, 64)])[0], [(10, 64)])

    def test_invalid_limits_and_metadata(self):
        for args in ((0, 1, 8), (64, 0, 8), (64, 1, 0), (True, 1, 8),
                     (64, 257, 8), (64, 1, 1.5)):
            with self.subTest(args=args), self.assertRaises(ValueError):
                ReclaimPlanner(*args)
        for runs in ([(1, -1)], [(1, True)], [(1, 8), (1, 8)], [(True, 8)], [(1.5, 8)]):
            with self.subTest(runs=runs), self.assertRaises(ValueError):
                ReclaimPlanner(64).plan(runs)


class AdapterWiringTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {"GSVECTOR_PG_DSN": "dbname=not_connected",
                                          "GSVECTOR_ARTIFACTS": self.tmp.name})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.tmp.cleanup()

    def adapter(self, enabled=True, **params):
        adapter = Gsvector("euclidean", {"compaction": "tiered", "compact_runs": 4,
                                         "reclaim_max_rows": 64 if enabled else 0,
                                         "reclaim_cooldown": 3, **params})
        self.addCleanup(adapter.done)
        adapter.conn = MagicMock()
        adapter.schema = "test_reclaim"
        adapter.index_ready = True
        adapter._runs = MagicMock(return_value=[("1", 64, 1)])
        adapter._record = MagicMock()
        adapter.conn.execute.return_value.fetchone.return_value = ({
            "receipt": 0, "outcome": "insufficient_reclaimable",
            "input_rows": 64, "counted_survivors": 64},)
        return adapter

    @staticmethod
    def reclaim_calls(adapter):
        return [c for c in adapter.conn.execute.call_args_list
                if isinstance(c.args[0], str) and "gsvector_reclaim_run" in c.args[0]]

    def test_default_off_and_insert_only_are_inert(self):
        for enabled, dirty in ((False, True), (True, False)):
            adapter = self.adapter(enabled)
            adapter._reclaim_dirty = dirty
            adapter._maintenance()
            self.assertEqual(self.reclaim_calls(adapter), [])
            self.assertNotIn("reclaim_calls", adapter.maintenance_totals)

    def test_single_run_gate_bypassed_but_cooldown_respected(self):
        adapter = self.adapter()
        adapter._reclaim_dirty = True
        adapter._maintenance()
        self.assertEqual(len(self.reclaim_calls(adapter)), 1)
        self.assertEqual(self.reclaim_calls(adapter)[0].args[1], ("test_reclaim.ann_idx", 1, 64, 1))
        self.assertEqual(adapter.maintenance_totals["reclaim_calls"], 1)
        adapter._maintenance()
        adapter._maintenance()
        self.assertEqual(len(self.reclaim_calls(adapter)), 1)
        adapter._maintenance()
        self.assertEqual(len(self.reclaim_calls(adapter)), 2)

    def test_backoff_feedback_and_retry_telemetry(self):
        adapter = self.adapter(reclaim_backoff_max=12)
        adapter._reclaim_dirty = True
        adapter._maintenance()
        self.assertEqual(adapter._record.call_args.kwargs["outcomes"][0]["retry_after_rounds"], 6)
        for _ in range(5):
            adapter._maintenance()
        self.assertEqual(len(self.reclaim_calls(adapter)), 1)
        adapter._maintenance()
        self.assertEqual(len(self.reclaim_calls(adapter)), 2)
        self.assertEqual(adapter._record.call_args.kwargs["outcomes"][0]["retry_after_rounds"], 12)

    def test_grouped_rerank_option_and_persistent_reset(self):
        adapter = self.adapter()
        for value in (1, "on", None):
            with self.assertRaises(ValueError):
                adapter._set_query_arguments({"grouped_rerank": value}, reset_workload=False)
        adapter._set_query_arguments({"grouped_rerank": True}, reset_workload=False)
        adapter._apply_random_io(adapter.conn)
        self.assertIn(
            (("SELECT set_config('gsvector.grouped_rerank',%s,false)", ("on",)),),
            adapter.conn.execute.call_args_list)
        adapter._set_query_arguments({}, reset_workload=False)
        adapter._apply_random_io(adapter.conn)
        adapter.conn.execute.assert_called_with("RESET gsvector.grouped_rerank")

    def test_federate_budget_validation_and_reset(self):
        adapter = self.adapter()
        for value in (True, None, "2", 0, 8.01, float("inf"), float("nan"), 10**400):
            with self.subTest(value=value), self.assertRaises(ValueError):
                adapter._set_query_arguments({"federate_k": value}, reset_workload=False)
        adapter._set_query_arguments({"federate_k": 2}, reset_workload=False)
        adapter._apply_random_io(adapter.conn)
        adapter.conn.execute.assert_called_with(
            "SELECT set_config('gsvector.federate_k',%s,false)", ("2",))
        adapter._set_query_arguments({}, reset_workload=False)
        adapter._apply_random_io(adapter.conn)
        adapter.conn.execute.assert_called_with("RESET gsvector.federate_k")

    def test_retire_only_configuration_requires_all_rows_reclaimable(self):
        path = (Path(__file__).resolve().parents[1] /
                "neurips23/streaming/gsvector/config.yaml")
        group = yaml.safe_load(path.read_text())["msturing-30M-clustered"][
            "gsvector-streaming-retire-only"]["run-groups"]["experimental-retirement"]
        adapter = Gsvector("euclidean", json.loads(group["args"])[0])
        self.addCleanup(adapter.done)
        adapter.set_query_arguments(json.loads(group["query-args"])[0])
        for rows in (1, 64, 1000000):
            self.assertEqual(adapter.reclaim_threshold.minimum(rows), rows)
        self.assertEqual(adapter.reclaim_max_rows, 1000000)
        self.assertEqual(adapter.reclaim_max_runs, 2)
        self.assertEqual(adapter.query_args["federate_k"], 2.0)

    def test_failure_diagnostic_configs_change_one_factor_only(self):
        path = (Path(__file__).resolve().parents[1] /
                "neurips23/streaming/gsvector/config.yaml")
        configs = yaml.safe_load(path.read_text())["msturing-30M-clustered"]
        def arguments(name):
            group, = configs[name]["run-groups"].values()
            return json.loads(group["args"])[0], json.loads(group["query-args"])[0]
        base_index, base_query = arguments("gsvector-streaming-retire-only")
        for name in ("gsvector-streaming-retire-reconnect", "gsvector-streaming-retire-codeoff"):
            index, query = arguments(name)
            adapter = Gsvector("euclidean", index)
            self.addCleanup(adapter.done)
            adapter.set_query_arguments(query)
            if name.endswith("reconnect"):
                self.assertFalse(query.pop("reuse_connections"))
                expected = dict(base_query)
                expected.pop("reuse_connections")
                self.assertEqual(query, expected)
                self.assertEqual(index, base_index)
            else:
                self.assertFalse(index.pop("build_neighbor_codes"))
                self.assertEqual(index, base_index)
                self.assertEqual(query, base_query)

    def test_fraction_threshold_is_passed_and_recorded(self):
        for absolute, expected in ((1, 13), (20, 20)):
            adapter = self.adapter(reclaim_min_dead_fraction=.2, reclaim_min_dead=absolute)
            adapter._reclaim_dirty = True
            adapter._maintenance()
            self.assertEqual(self.reclaim_calls(adapter)[0].args[1],
                             ("test_reclaim.ann_idx", 1, 64, expected))
            outcome = adapter._record.call_args.kwargs["outcomes"][0]
            self.assertEqual(outcome["min_reclaimable_rows"], expected)
            self.assertEqual(outcome["outcome"], "insufficient_reclaimable")
            self.assertEqual(adapter.maintenance_totals["reclaim_calls"], 1)
            adapter._maintenance()
            self.assertEqual(len(self.reclaim_calls(adapter)), 1)

    def test_error_propagation_and_oversized_reporting(self):
        adapter = self.adapter()
        adapter._reclaim_dirty = True
        adapter._runs.return_value = [("1", 65, 1)]
        adapter._maintenance()
        self.assertEqual(self.reclaim_calls(adapter), [])
        self.assertEqual(adapter._record.call_args.kwargs["oversized_runs"],
                         [{"run_id": 1, "input_rows": 65}])
        adapter._runs.return_value = [("1", 64, 1)]
        def fail_reclaim(statement, args=None):
            if "gsvector_reclaim_run" in statement:
                raise RuntimeError("maintenance conflict")
            return MagicMock()
        adapter.conn.execute.side_effect = fail_reclaim
        with self.assertRaisesRegex(RuntimeError, "maintenance conflict"):
            adapter._maintenance()
        self.assertEqual(adapter.maintenance_totals["reclaim_calls"], 1)

    def test_old_server_rejected_before_workload_creation(self):
        adapter = self.adapter()
        connection = MagicMock()
        connection.execute.return_value.fetchone.return_value = (False,)
        adapter._new_workload = MagicMock()
        with patch("neurips23.streaming.gsvector.module.psycopg.connect", return_value=connection):
            with self.assertRaisesRegex(ValueError, "upgrade explicitly"):
                adapter.setup("float32", 64, 32)
        connection.close.assert_called_once()
        adapter._new_workload.assert_not_called()

    def test_normal_minor_output_is_not_immediately_reclaimed_again(self):
        adapter = self.adapter()
        adapter._reclaim_dirty = True
        adapter._runs.side_effect = [[(str(i), 16, 1) for i in range(1, 5)], [("5", 64, 1)]]
        receipts = iter((5, 0))
        def execute(statement, args=None):
            cursor = MagicMock()
            if "gsvector_compact" in statement:
                cursor.fetchone.return_value = (next(receipts),)
            return cursor
        adapter.conn.execute.side_effect = execute
        adapter._maintenance()
        self.assertEqual(adapter.maintenance_totals["compact_calls"], 2)
        self.assertEqual(self.reclaim_calls(adapter), [])
        self.assertEqual(adapter.reclaim_planner.due, {5: 4})

    def test_new_parameter_validation(self):
        for args in ({"reclaim_max_rows": 64}, {"reclaim_max_rows": True},
                     {"reclaim_min_dead": 0}, {"reclaim_max_runs": "1"},
                     {"reclaim_cooldown": 0}, {"reclaim_max_rows": -1},
                     {"reclaim_min_dead_fraction": True},
                     {"reclaim_min_dead_fraction": float("nan")},
                     {"reclaim_min_dead_fraction": "0.2"},
                     {"reclaim_backoff_max": True}, {"reclaim_backoff_max": -1},
                     {"reclaim_backoff_max": 4}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                Gsvector("euclidean", args)


if __name__ == "__main__":
    unittest.main()
