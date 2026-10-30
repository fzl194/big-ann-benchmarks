import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import numpy as np
from benchmark.algorithms.pg_dense import PGDenseANN
from neurips23.streaming.gsvector.module import Gsvector


class RefineU8SimdTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, GSVECTOR_PG_DSN='dbname=unused', GSVECTOR_ARTIFACTS=self.tmp.name)
        self.env.start()
        self.algo = Gsvector('euclidean', {})

    def tearDown(self):
        self.algo.done()
        self.env.stop()
        self.tmp.cleanup()

    def configure(self, **args):
        self.algo.set_query_arguments({'search_l': 128, 'threads': 1, **args})

    def test_validate_and_reset(self):
        for invalid in (0, 1, 'on', None):
            with self.assertRaises(ValueError): self.configure(refine_u8_simd=invalid)
        self.configure(refine_sq8=True, refine_u8_simd=True)
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = ('bool',)
        self.algo.conn = conn
        self.algo._apply_refine_u8_simd(conn)
        conn.execute.assert_any_call("SELECT set_config('gsvector.refine_u8_simd',%s,false)", ('on',))
        self.configure()
        self.algo._apply_refine_u8_simd(conn)
        conn.execute.assert_any_call("SELECT set_config('gsvector.refine_u8_simd',%s,false)", ('off',))

    def test_older_server_and_worker_reuse(self):
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = None
        self.algo.conn = conn
        self.configure()
        self.algo._apply_refine_u8_simd(conn)
        conn.execute.assert_not_called()
        self.configure(refine_u8_simd=True)
        with self.assertRaises(RuntimeError): self.algo._apply_refine_u8_simd(conn)
        owner, worker = MagicMock(), MagicMock()
        worker.execute.return_value.fetchone.return_value = ('bool',)
        self.algo.conn = owner
        self.configure(refine_u8_simd=True, reuse_connections=True)
        self.algo._apply_refine_u8_simd(worker)
        self.configure(reuse_connections=True)
        self.algo._apply_refine_u8_simd(worker)
        worker.execute.assert_any_call("SELECT set_config('gsvector.refine_u8_simd',%s,false)", ('off',))

    def test_single_query_switch_updates_connection(self):
        self.algo.done()
        self.algo = PGDenseANN('euclidean', {})
        self.algo.schema='owned'; self.algo.dim=2; self.algo.live_count=1
        self.algo.vector_oid=1234; self.algo.keep_schema=True
        self.algo._runs=MagicMock(return_value=[]); self.algo._record=MagicMock()
        conn=MagicMock()
        def execute(statement,*args):
            result=MagicMock()
            if statement.startswith('SELECT vartype'):
                result.fetchone.return_value=('bool',)
            elif statement.startswith('EXPLAIN '):
                result.fetchone.return_value=([{'Plan':{'Node Type':'Index Scan','Index Name':'ann_idx'}}],)
            else: result.fetchall.return_value=[(17,)]
            return result
        conn.execute.side_effect=execute; self.algo.conn=conn
        for simd in (True,False,None):
            args={'refine_sq8':True}
            if simd is not None: args['refine_u8_simd']=simd
            self.configure(**args)
            self.algo.query(np.array([1,2],dtype=np.float32),1)
        writes=[call.args[1] for call in conn.execute.call_args_list
                if call.args[0]=="SELECT set_config('gsvector.refine_u8_simd',%s,false)"]
        self.assertEqual(writes,[('on',),('off',),('off',)])

if __name__ == '__main__':
    unittest.main()
