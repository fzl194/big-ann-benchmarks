import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from neurips23.streaming.gsvector.module import Gsvector
from benchmark.algorithms.pg_dense import PGDenseANN
import numpy as np

class ScoreLimitTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, GSVECTOR_PG_DSN='dbname=unused',
                              GSVECTOR_ARTIFACTS=self.tmp.name)
        self.env.start()
        self.algo = Gsvector('euclidean', {})

    def tearDown(self):
        self.algo.done()
        self.env.stop()
        self.tmp.cleanup()

    def configure(self, **kwargs):
        self.algo.set_query_arguments({'search_l':1280, 'threads':1, **kwargs})

    def test_bounds_and_required_refinement(self):
        for value in (-1,32769,True,1.5,'4'):
            with self.assertRaises(ValueError): self.configure(refine_sq8=True,refine_score_limit=value)
        with self.assertRaises(ValueError): self.configure(refine_score_limit=10)
        with self.assertRaises(ValueError):
            self.configure(refine_sq8=True,refine_score_limit=10,refine_expanded_pool=10)
        self.configure(refine_sq8=True,refine_score_limit=32768)

    def test_new_and_reused_connection_resets(self):
        conn=MagicMock()
        conn.execute.return_value.fetchone.return_value=('integer',)
        self.algo.conn=conn
        self.configure(refine_sq8=True,refine_score_limit=256)
        self.algo._apply_refine_score_limit(conn)
        conn.execute.assert_any_call("SELECT set_config('gsvector.refine_score_limit',%s,false)",('256',))
        self.configure()
        self.algo._apply_refine_score_limit(conn)
        conn.execute.assert_any_call("SELECT set_config('gsvector.refine_score_limit',%s,false)",('0',))

    def test_old_server_omission_and_explicit_failure(self):
        conn=MagicMock()
        conn.execute.return_value.fetchone.return_value=None
        self.algo.conn=conn
        self.configure()
        self.algo._apply_refine_score_limit(conn)
        conn.execute.assert_not_called()
        self.configure(refine_sq8=True,refine_score_limit=1)
        with self.assertRaises(RuntimeError): self.algo._apply_refine_score_limit(conn)

    def test_reused_worker_resets_omitted_pool_and_old_server_is_safe(self):
        owner=MagicMock(); worker=MagicMock()
        worker.execute.return_value.fetchone.return_value=('integer',)
        self.algo.conn=owner
        self.configure(refine_sq8=True,refine_score_limit=99,reuse_connections=True)
        self.algo._apply_refine_score_limit(worker)
        self.configure(reuse_connections=True)
        self.algo._apply_refine_score_limit(worker)
        worker.execute.assert_any_call(
            "SELECT set_config('gsvector.refine_score_limit',%s,false)",('0',))
        old=MagicMock(); old.execute.return_value.fetchone.return_value=None
        self.algo._apply_refine_score_limit(old)
        self.assertFalse(any("set_config" in c.args[0] for c in old.execute.call_args_list))

    def test_single_query_state_tracks_pool_changes(self):
        self.algo.done()
        self.algo=PGDenseANN('euclidean', {})
        self.algo.schema='owned'; self.algo.dim=2; self.algo.live_count=1
        self.algo.vector_oid=1234; self.algo.keep_schema=True
        self.algo._runs=MagicMock(return_value=[]); self.algo._record=MagicMock()
        conn=MagicMock()
        def execute(statement,*args):
            result=MagicMock()
            if statement.startswith("SELECT vartype"):
                result.fetchone.return_value=('integer' if args[0][0].endswith('score_limit') else 'bool',)
            elif statement.startswith("EXPLAIN "):
                result.fetchone.return_value=([{'Plan':{'Node Type':'Index Scan','Index Name':'ann_idx'}}],)
            else: result.fetchall.return_value=[(17,)]
            return result
        conn.execute.side_effect=execute; self.algo.conn=conn
        for pool in (64,128,0):
            self.configure(refine_sq8=True,refine_score_limit=pool)
            self.algo.query(np.array([1,2],dtype=np.float32),1)
        writes=[x.args[1] for x in conn.execute.call_args_list
                if x.args[0]=="SELECT set_config('gsvector.refine_score_limit',%s,false)"]
        self.assertEqual(writes,[('64',),('128',),('0',)])

if __name__ == '__main__':
    unittest.main()
