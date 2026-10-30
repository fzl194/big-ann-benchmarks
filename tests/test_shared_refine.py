import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from benchmark.algorithms.pg_dense import PGDenseANN
from neurips23.streaming.gsvector.module import Gsvector


GUC = 'gsvector.refine_sq8'
SET_SQL = "SELECT set_config('gsvector.refine_sq8',%s,false)"


class SharedRefineArguments(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {
            'GSVECTOR_PG_DSN': 'dbname=not_connected',
            'GSVECTOR_ARTIFACTS': self.temp.name,
        })
        self.env.start()
        self.algo = Gsvector('euclidean', {})

    def tearDown(self):
        self.algo.done()
        self.env.stop()
        self.temp.cleanup()

    def configure(self, **args):
        self.algo.set_query_arguments({'search_l': 1280, 'threads': 1, **args})

    @staticmethod
    def connection(kind='bool'):
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = None if kind is None else (kind,)
        return conn

    @staticmethod
    def writes(conn):
        return [call.args for call in conn.execute.call_args_list
                if isinstance(call.args[0], str) and call.args[0] == SET_SQL]

    def test_new_mode_is_query_only_and_does_not_change_search_budget(self):
        self.configure(refine_sq8=True, max_k=20, federate_k=1.0)
        self.assertIs(self.algo.query_args['refine_sq8'], True)
        self.assertEqual(self.algo.query_args['search_l'], 1280)
        self.assertEqual(self.algo.query_args['max_k'], 20)
        with self.assertRaisesRegex(ValueError, 'unknown index parameters'):
            Gsvector('euclidean', {'refine_sq8': True})

    def test_refinement_requires_a_boolean(self):
        for value in (0, 1, None, 'on', [], np.bool_(True)):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, 'refine_sq8 must be boolean'):
                self.configure(refine_sq8=value)

    def test_full_raw_and_compressed_modes_cannot_both_be_enabled(self):
        with self.assertRaisesRegex(ValueError, 'mutually exclusive'):
            self.configure(refine_sq8=True, pq_rerank_frontier=True)
        for mode in (False, True):
            self.configure(refine_sq8=mode, pq_rerank_frontier=not mode)

    def test_explicit_modes_require_registered_boolean_before_setting(self):
        for enabled in (False, True):
            self.configure(refine_sq8=enabled)
            conn = self.connection()
            self.algo.conn = conn
            self.algo._apply_random_io(conn)
            conn.execute.assert_any_call('SELECT vartype FROM pg_settings WHERE name = %s', (GUC,))
            self.assertEqual(self.writes(conn), [(SET_SQL, ('on' if enabled else 'off',))])

    def test_old_or_wrong_type_server_rejects_explicit_mode(self):
        for kind in (None, 'integer', 'string'):
            self.configure(refine_sq8=True)
            conn = self.connection(kind)
            self.algo.conn = conn
            with self.assertRaisesRegex(RuntimeError, 'matching.*extension'):
                self.algo._apply_random_io(conn)
            self.assertEqual(self.writes(conn), [])

    def test_omitted_mode_keeps_old_server_compatibility(self):
        self.configure()
        conn = self.connection(None)
        self.algo.conn = conn
        self.algo._apply_random_io(conn)
        self.assertFalse(any(GUC in str(call) for call in conn.execute.call_args_list))

    def test_reused_worker_connections_reset_after_omission(self):
        conns = [self.connection(), self.connection()]
        self.configure(refine_sq8=True, reuse_connections=True)
        for conn in conns:
            self.algo._apply_random_io(conn)
        self.configure(reuse_connections=True)
        for conn in conns:
            conn.execute.reset_mock()
            self.algo._apply_random_io(conn)
            self.assertEqual(self.writes(conn), [(SET_SQL, ('off',))])

    def dense(self):
        self.algo.done()
        self.algo = PGDenseANN('euclidean', {})
        self.algo.schema = 'owned_test'
        self.algo.dim = 2
        self.algo.live_count = 1
        self.algo.vector_oid = 1234
        self.algo.keep_schema = True
        self.algo._runs = MagicMock(return_value=[])
        self.algo._record = MagicMock()
        conn = self.connection()
        cursor = MagicMock()
        cursor.fetchone.return_value = ('bool',)
        cursor.fetchall.return_value = [(17,)]
        plan = MagicMock()
        plan.fetchone.return_value = ([{'Plan': {'Node Type': 'Index Scan', 'Index Name': 'ann_idx'}}],)
        conn.execute.side_effect = lambda sql, *args: plan if isinstance(sql, str) and sql.startswith('EXPLAIN ') else cursor
        self.algo.conn = conn
        return conn

    def test_actual_single_query_mode_changes_invalidate_cached_session_state(self):
        conn = self.dense()
        for args in ({'refine_sq8': False}, {'refine_sq8': True}, {}):
            self.configure(**args)
            np.testing.assert_array_equal(self.algo.query(np.asarray([1, 2], dtype=np.float32), 1), [17])
        self.assertEqual(self.writes(conn), [
            (SET_SQL, ('off',)), (SET_SQL, ('on',)), (SET_SQL, ('off',))])


if __name__ == '__main__':
    unittest.main()
