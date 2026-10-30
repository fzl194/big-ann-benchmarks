import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from benchmark.algorithms.pg_filtered import PGFilteredANN


class FilterBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            "GSVECTOR_PG_DSN": "dbname=not_connected",
            "GSVECTOR_ARTIFACTS": self.temp.name,
        })
        self.env.start()
        self.adapter = PGFilteredANN("euclidean", {})
        self.adapter.nb = 1000

    def tearDown(self):
        self.adapter.done()
        self.env.stop()
        self.temp.cleanup()

    def configure(self, **args):
        self.adapter.set_query_arguments({"filter_strategy": "bounded",
                                          "ann_candidates": 128, **args})

    def test_budget_is_explicit_and_legacy_remains_available(self):
        self.assertEqual(self.adapter.filter_strategy, "legacy")
        self.configure()
        self.assertEqual(self.adapter.ann_candidates, 128)
        for args in ({"filter_strategy": "unknown"}, {"ann_candidates": True},
                     {"ann_candidates": 0}, {"ann_candidates": 8193},
                     {"refine_sq8": "on"}):
            with self.subTest(args=args), self.assertRaises(ValueError):
                self.configure(**args)

    def test_build_budget_and_precision_are_validated(self):
        good = {"build_neighbor_code_bytes": 512, "pq_m": 24,
                "build_raw_cache_mb": 128, "build_gather_threads": 4}
        algo = PGFilteredANN("euclidean", good)
        self.assertEqual(algo.pq_m, 24)
        self.assertEqual(algo.build_options["build_neighbor_code_bytes"], 512)
        for name, value in (("pq_m", True), ("pq_m", -1),
                            ("build_neighbor_code_bytes", -2),
                            ("build_raw_cache_mb", 65537),
                            ("build_gather_threads", 0)):
            with self.subTest(name=name), self.assertRaises(ValueError):
                PGFilteredANN("euclidean", {name: value})

    def test_bounded_cutoff_is_not_silently_expanded_with_dataset_size(self):
        self.adapter.nb=10_000_000
        self.configure(exact_cutoff=2048)
        self.assertEqual(self.adapter._effective_cutoff(10),2048)
        self.adapter.set_query_arguments({})
        self.assertEqual(self.adapter._effective_cutoff(10),24415)

    def test_refinement_cannot_build_codes_inside_query_timing(self):
        self.configure(refine_sq8=True)
        with self.assertRaisesRegex(RuntimeError,"prepare_refine"):
            self.adapter.filtered_query(None,None,10)

    def test_build_applies_integer_options_and_prepares_codes_before_queries(self):
        algo=PGFilteredANN("euclidean",{"pq_m":24,"build_neighbor_code_bytes":0,
                                         "build_raw_cache_mb":128,"build_gather_threads":4,
                                         "prepare_refine":True})
        conn=MagicMock()
        def fitted(_name):
            algo.conn=conn
            algo.dim=192
            algo.nb=96
        conn.execute.return_value.fetchone.return_value=("integer",)
        with patch("benchmark.algorithms.pg_exact.PGFilterExact.fit",side_effect=fitted):
            algo.fit("fixture")
        self.assertTrue(algo.refine_prepared)
        self.assertIn("prepare_refine_s",algo.build_times)
        conn.execute.assert_any_call("SELECT set_config(%s,%s,false)",
                                     ("gsvector.build_neighbor_code_bytes","0"))
        conn.execute.assert_any_call("SELECT set_config(%s,%s,false)",
                                     ("gsvector.build_gather_threads","4"))
        conn.execute.assert_any_call("SELECT prepare_refine('ann_idx'::regclass)")

    def test_build_rejects_missing_guc_and_incompatible_precision(self):
        for options,dim in (({"build_neighbor_code_bytes":0},192),({"pq_m":7},192)):
            algo=PGFilteredANN("euclidean",options)
            def fitted(_name):
                algo.conn=MagicMock()
                algo.conn.execute.return_value.fetchone.return_value=None
                algo.dim=dim
                algo.nb=96
            with patch("benchmark.algorithms.pg_exact.PGFilterExact.fit",side_effect=fitted):
                with self.assertRaises((RuntimeError,ValueError)):
                    algo.fit("fixture")

    @staticmethod
    def connection(ids, ann=(), exact=()):
        conn = MagicMock()
        def execute(statement, params=None):
            cursor = MagicMock()
            if statement.startswith("SELECT id FROM base WHERE tags") and "ORDER BY" not in statement:
                cursor.__iter__.return_value = iter((i,) for i in ids)
            elif statement.startswith("EXPLAIN"):
                cursor.fetchone.return_value = ([{"Plan": {"Node Type": "Index Scan",
                                                          "Index Name": "ann_idx"}}],)
            elif "ORDER BY embedding" in statement:
                values = exact if "embedding <-> %s,id" in statement else ann
                cursor.__iter__.return_value = iter((i,) for i in values)
            return cursor
        conn.execute.side_effect = execute
        return conn

    def run_bounded(self, conn, tags=(3,), cutoff=2):
        self.configure()
        stats = self.adapter._new_filter_stats()
        result = self.adapter._bounded_query(conn, list(tags), b"query", 2, cutoff, stats)
        return result, stats

    def test_small_set_reuses_ids_and_preserves_label_predicate(self):
        conn = self.connection([7, 9], exact=[9, 7])
        result, stats = self.run_bounded(conn)
        self.assertEqual(result, [9, 7])
        self.assertEqual(stats["small_exact_queries"], 1)
        self.assertEqual(stats["reused_eligible_ids"], 2)
        ranking = [(c.args[0], c.args[1]) for c in conn.execute.call_args_list
                   if "ORDER BY embedding" in c.args[0]]
        self.assertEqual(len(ranking), 1)
        self.assertIn("id = ANY", ranking[0][0])
        self.assertIn("tags @>", ranking[0][0])
        self.assertEqual(ranking[0][1], ([7, 9], [3], b"query", 2))

    def test_empty_set_does_not_search_or_score(self):
        conn = self.connection([])
        result, stats = self.run_bounded(conn)
        self.assertEqual(result, [])
        self.assertEqual(stats["ann_attempts"], 0)
        self.assertFalse(any("ORDER BY" in c.args[0] for c in conn.execute.call_args_list))

    def test_large_set_uses_one_ann_attempt_and_never_ranks_probe_subset(self):
        conn = self.connection([1, 2, 3], ann=[90, 91])
        result, stats = self.run_bounded(conn)
        self.assertEqual(result, [90, 91])
        self.assertEqual(stats["ann_attempts"], 1)
        self.assertEqual(stats["max_candidates_used"], 128)
        self.assertEqual(stats["reused_eligible_ids"], 0)
        self.assertFalse(any("id = ANY" in c.args[0] for c in conn.execute.call_args_list))

    def test_short_ann_result_falls_back_once_without_retry(self):
        conn = self.connection([1, 2, 3], ann=[90], exact=[90, 91])
        result, stats = self.run_bounded(conn)
        self.assertEqual(result, [90, 91])
        self.assertEqual(stats["ann_attempts"], 1)
        self.assertEqual(stats["fallback_exact_queries"], 1)
        self.assertEqual(sum("ORDER BY embedding" in c.args[0] and not c.args[0].startswith("EXPLAIN")
                             for c in conn.execute.call_args_list), 2)

    def test_empty_filter_skips_label_probe(self):
        conn = self.connection([], ann=[4, 5])
        result, stats = self.run_bounded(conn, tags=())
        self.assertEqual(result, [4, 5])
        self.assertEqual(stats["eligible_probe_queries"], 0)

    def test_zero_cutoff_skips_probe_and_small_corpus_clamps_budget(self):
        self.adapter.nb=8
        conn=self.connection([],ann=[4,5])
        result,stats=self.run_bounded(conn,cutoff=0)
        self.assertEqual(result,[4,5])
        self.assertEqual(stats["eligible_probe_queries"],0)
        self.assertEqual(stats["max_candidates_used"],8)

    def test_unexpected_query_plan_is_not_accepted_as_ann(self):
        conn=self.connection([1,2,3])
        original=conn.execute.side_effect
        def execute(statement,params=None):
            result=original(statement,params)
            if statement.startswith("EXPLAIN"):
                result.fetchone.return_value=([{"Plan":{"Node Type":"Seq Scan"}}],)
            return result
        conn.execute.side_effect=execute
        with self.assertRaisesRegex(RuntimeError,"native index"):
            self.run_bounded(conn)

    def test_sql_errors_are_not_disguised_as_exact_fallback(self):
        conn = MagicMock()
        conn.execute.side_effect = RuntimeError("database error")
        with self.assertRaisesRegex(RuntimeError, "database error"):
            self.run_bounded(conn)


if __name__ == "__main__":
    unittest.main()
