import os
import tempfile
import unittest
from unittest.mock import patch, MagicMock
from concurrent.futures import Future

import numpy as np
import psycopg
from psycopg import sql

from neurips23.streaming.gsvector.adaptive import AdaptiveGsvector, ReindexPlanner, query_budget
from neurips23.streaming.gsvector.module import Gsvector, QueryCardinalityError, gather_query_chunks


class PolicyTests(unittest.TestCase):
    def test_non_cardinality_errors_are_not_hidden(self):
        futures=[Future(),Future(),Future()]
        futures[0].set_result(np.zeros((3,10)))
        futures[1].set_exception(QueryCardinalityError(9,10,2))
        futures[2].set_exception(RuntimeError("corruption"))
        with self.assertRaisesRegex(RuntimeError,"corruption"):
            gather_query_chunks(futures)
        futures=futures[:2]
        with self.assertRaises(QueryCardinalityError) as error:
            gather_query_chunks(futures)
        self.assertEqual(error.exception.completed_queries_total,5)

    def test_repair_is_bounded_and_all_work_is_timed(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ,{
                "GSVECTOR_PG_DSN":"dbname=not_connected","GSVECTOR_ARTIFACTS":directory}):
            adapter=AdaptiveGsvector("euclidean",{"query_run_floor":8,"query_rebuild_retries":1})
            adapter.schema="fixture";adapter.live_count=100
            adapter.query_args={"search_l":200,"max_k":100,"threads":1}
            original=dict(adapter.query_args)
            adapter._runs=MagicMock(return_value=[(1,200,1)])
            adapter._perform_reindex=MagicMock()
            calls=[]
            def query(self,X,k):
                calls.append(dict(self.query_args))
                if len(calls)==1:
                    raise QueryCardinalityError(9,10,5)
                self._record("search",queries=len(X),k=k,elapsed_s=1,
                             query_args=dict(self.query_args),runs=self._runs())
            with patch.object(Gsvector,"query",query), \
                 patch("neurips23.streaming.gsvector.adaptive.time.perf_counter",side_effect=[10,42]):
                adapter.query(np.zeros((10,32)),10)
            adapter._perform_reindex.assert_called_once()
            self.assertEqual(adapter.query_args,original)
            self.assertEqual(adapter.events[-1]["elapsed_s"],32)
            self.assertEqual(adapter.events[-1]["executed_ann_queries"],15)
            self.assertEqual([x["status"] for x in adapter.events[-1]["query_attempts"]],
                             ["shortfall","pass"])
            adapter._perform_reindex.reset_mock()
            with patch.object(Gsvector,"query",side_effect=QueryCardinalityError(9,10,1)):
                with self.assertRaises(QueryCardinalityError):
                    adapter.query(np.zeros((10,32)),10)
            adapter._perform_reindex.assert_called_once()
            self.assertEqual(adapter.query_args,original)
            adapter.done()

    def test_threshold_cooldown_and_reset(self):
        planner = ReindexPlanner(2, 100, 3)
        self.assertFalse(planner.plan(200, 101))
        self.assertTrue(planner.plan(200, 100))
        self.assertFalse(planner.plan(200, 50))
        self.assertFalse(planner.plan(200, 50))
        self.assertTrue(planner.plan(200, 50))
        self.assertFalse(planner.plan(200, 0))
        planner.reset()
        self.assertTrue(planner.plan(200, 100))
        self.assertFalse(ReindexPlanner().plan(10000000, 1))

    def test_invalid_policy_values(self):
        for ratio in (True, None, -1, 1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                ReindexPlanner(ratio)
        for value in (True, 0, -1, 1.5):
            with self.assertRaises(ValueError):
                ReindexPlanner(2, value)
            with self.assertRaises(ValueError):
                ReindexPlanner(2, 100, value)

    def test_budget_floor_changes_only_small_source_counts(self):
        base = {"search_l": 200, "max_k": 100, "threads": 32, "federate_k": 2}
        for runs, L, k in ((0,1600,800),(1,1600,800),(2,800,400),
                           (4,400,200),(8,200,100),(40,200,100)):
            result = query_budget(base, runs, 8)
            self.assertEqual((result["search_l"],result["max_k"]), (L,k))
            self.assertEqual(result["threads"], 32)
        self.assertEqual(base["search_l"],200)
        self.assertEqual(query_budget(base,1,0),base)
        with self.assertRaises(ValueError):
            query_budget({"search_l":200},1,8)
        for floor in (True,-1,65,1.5):
            with self.assertRaises(ValueError):
                query_budget(base,1,floor)


@unittest.skipUnless(os.environ.get("GSVECTOR_PG_DSN"), "requires an owned test database")
class AdaptiveIntegrationTests(unittest.TestCase):
    def test_shortfall_rebuild_retries_full_query_and_keeps_accounting(self):
        vectors=np.random.default_rng(118).normal(size=(256,32)).astype("float32")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "GSVECTOR_ARTIFACTS":directory,"GSVECTOR_KEEP_SCHEMA":"0"}):
            adapter=AdaptiveGsvector("euclidean",{
                "R":8,"L":32,"build_threads":1,"min_run_rows":0,
                "query_rebuild_retries":1,"reconnect_writer":True})
            try:
                adapter.setup("float32",256,32)
                args={"search_l":8,"max_k":8,"threads":1,"federate_k":1,
                      "reuse_connections":True}
                adapter.set_query_arguments(args)
                adapter.insert(vectors,np.arange(256))
                adapter.query(vectors[:1],8)
                removed=adapter.res[0].copy()
                adapter.delete(removed)
                adapter.query(vectors[:1],5)
                self.assertEqual(adapter.res.shape,(1,5))
                self.assertFalse(set(adapter.res[0])&set(removed))
                event=adapter.events[-1]
                self.assertEqual(event["executed_ann_queries"],2)
                self.assertEqual(event["maintenance_reindexes"][0]["reason"],"query_shortfall")
                self.assertEqual(event["maintenance_totals"]["reindex_calls"],1)
                self.assertEqual(adapter.search_number,2)
                self.assertEqual(adapter.query_args,args)
            finally:adapter.done()

    def fixture(self, directory):
        return AdaptiveGsvector("euclidean", {
            "R":8,"L":32,"build_threads":2,"batch_rows":128,"min_run_rows":0,
            "vacuum_memtables":True,"reconnect_writer":True,"compact_max_rows":128,
            "reindex_ratio":2,"reindex_min_rows":64,"reindex_cooldown":2,
            "query_run_floor":8,
        })

    def test_reindex_preserves_old_snapshot_then_removes_debt(self):
        vectors=np.random.default_rng(105).normal(size=(256,32)).astype("float32")
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "GSVECTOR_ARTIFACTS":directory,"GSVECTOR_KEEP_SCHEMA":"0"}):
            adapter=self.fixture(directory)
            try:
                adapter.setup("float32",256,32)
                args={"search_l":32,"max_k":10,"threads":2,"reuse_connections":True,"federate_k":2}
                adapter.set_query_arguments(args)
                adapter.insert(vectors,np.arange(256))
                with psycopg.connect(adapter.dsn,autocommit=True) as old:
                    old.execute(sql.SQL("SET search_path TO {},public").format(sql.Identifier(adapter.schema)))
                    old.execute("BEGIN ISOLATION LEVEL REPEATABLE READ")
                    old.execute("SELECT pg_current_snapshot()").fetchone()
                    adapter.delete(np.arange(192))
                    self.assertEqual(old.execute("SELECT count(*) FROM base").fetchone()[0],256)
                    self.assertEqual(adapter.maintenance_totals["reindex_calls"],1)
                    self.assertEqual(sum(r[1] for r in adapter._runs()),256)
                    old.execute("ROLLBACK")
                adapter._maintenance()
                adapter._maintenance()
                self.assertEqual(sum(r[1] for r in adapter._runs()),64)
                self.assertEqual(adapter.maintenance_totals["reindex_calls"],2)
                adapter.query(vectors[192:208],10)
                reference=adapter.res.copy()
                self.assertTrue((reference>=192).all())
                self.assertEqual(adapter.query_args,args)
                event=adapter.events[-1]
                self.assertEqual(event["query_args"]["search_l"],256)
                self.assertEqual(event["query_args"]["max_k"],80)
                self.assertEqual(event["budget_plan"]["run_count"],1)
                adapter.query(vectors[192:208],10)
                np.testing.assert_array_equal(adapter.res,reference)
                self.assertEqual(adapter.query_args,args)
                self.assertEqual(adapter.events[-1]["query_args"]["search_l"],256)
            finally:adapter.done()

    def test_query_failure_restores_nominal_budget(self):
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                "GSVECTOR_ARTIFACTS":directory,"GSVECTOR_KEEP_SCHEMA":"0"}):
            adapter=self.fixture(directory)
            try:
                adapter.setup("float32",64,32)
                args={"search_l":32,"max_k":10,"threads":1}
                adapter.set_query_arguments(args)
                with self.assertRaises(ValueError):
                    adapter.query(np.zeros((2,31),dtype="float32"),10)
                self.assertEqual(adapter.query_args,args)
            finally:adapter.done()


if __name__=="__main__":
    unittest.main()
