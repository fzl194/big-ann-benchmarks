import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import numpy as np
from psycopg.pq import TransactionStatus
from neurips23.streaming.gsvector.module import Gsvector


class SessionCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.env = patch.dict(os.environ, {'GSVECTOR_PG_DSN': 'dbname=fake',
                                          'GSVECTOR_ARTIFACTS': self.temp.name})
        self.env.start()
        self.adapter = Gsvector('euclidean', {})
        self.adapter.schema = 'fixture'
        self.adapter.dim, self.adapter.vector_oid, self.adapter.live_count = 4, 123, 100
        self.adapter.index_ready = True
        self.conn = self.connection()
        self.q = np.zeros((2, 4), dtype=np.float32)

    def tearDown(self):
        self.adapter._close_query_pool()
        self.env.stop()
        self.temp.cleanup()

    @staticmethod
    def connection():
        conn = MagicMock()
        conn.closed = False
        conn.broken = False
        conn.autocommit = True
        conn.info.transaction_status = TransactionStatus.IDLE
        conn.info.backend_pid = 24
        def execute(statement, *args):
            result = MagicMock()
            result.fetchone.return_value = ([],) if 'EXPLAIN (FORMAT JSON)' in str(statement) else None
            result.fetchall.return_value = [(i,) for i in range(3)]
            return result
        conn.execute.side_effect = execute
        return conn

    def configure(self, **changes):
        args = {'search_l': 60, 'threads': 1, 'reuse_connections': True,
                'cache_session_setup': True}
        args.update(changes)
        self.adapter._set_query_arguments(args, reset_workload=False)
        self.bind()

    def bind(self):
        if not self.adapter._query_connections:
            self.adapter._query_local.connection = self.conn
            self.adapter._query_connections = [self.conn]

    def setups(self):
        return sum('EXPLAIN (FORMAT JSON)' in str(c.args[0]) for c in self.conn.execute.call_args_list)

    @patch('neurips23.streaming.gsvector.module.has_index_plan', return_value=True)
    def test_cached_only_when_full_session_key_matches(self, plan):
        self.configure()
        self.adapter._query_chunk(self.q, 3)
        self.adapter._query_chunk(self.q, 3)
        self.assertEqual(self.setups(), 1)
        self.configure(search_l=80)
        self.adapter._query_chunk(self.q, 3)
        self.assertEqual(self.setups(), 2)
        self.adapter.vector_oid = 124
        self.adapter._query_chunk(self.q, 3)
        self.assertEqual(self.setups(), 3)
        self.conn.info.backend_pid = 25
        self.adapter._query_chunk(self.q, 3)
        self.assertEqual(self.setups(), 4)

    @patch('neurips23.streaming.gsvector.module.has_index_plan', return_value=True)
    def test_default_keeps_validating_each_batch(self, plan):
        self.adapter._set_query_arguments({'search_l': 60, 'threads': 1,
                                           'reuse_connections': True}, reset_workload=False)
        self.bind()
        self.adapter._query_chunk(self.q, 3)
        self.adapter._query_chunk(self.q, 3)
        self.assertEqual(self.setups(), 2)

    def test_opt_in_requires_connection_reuse(self):
        with self.assertRaisesRegex(ValueError, 'requires reuse_connections'):
            self.adapter._set_query_arguments({'cache_session_setup': True}, reset_workload=False)

    @patch('neurips23.streaming.gsvector.module.has_index_plan', return_value=True)
    def test_invalidation_closes_sessions(self, plan):
        self.configure()
        self.adapter._query_chunk(self.q, 3)
        self.adapter.invalidate_query_sessions()
        self.conn.close.assert_called_once()
        self.assertFalse(self.adapter._query_connections)

    @patch('neurips23.streaming.gsvector.module.has_index_plan', return_value=True)
    def test_nonidle_connection_is_never_cached(self, plan):
        self.configure()
        self.conn.info.transaction_status = TransactionStatus.INTRANS
        self.adapter._query_chunk(self.q, 3)
        self.adapter._query_chunk(self.q, 3)
        self.assertEqual(self.setups(), 2)

    @patch('neurips23.streaming.gsvector.module.has_index_plan', return_value=True)
    def test_schema_operator_and_maintenance_change_key(self, plan):
        self.configure()
        self.adapter._query_chunk(self.q, 3)
        self.adapter.schema = 'different'
        self.adapter._query_chunk(self.q, 3)
        self.adapter.operator = '<=>'
        self.adapter._query_chunk(self.q, 3)
        self.adapter._mark_query_sessions_stale()
        self.adapter._query_chunk(self.q, 3)
        self.assertEqual(self.setups(), 4)

    @patch('neurips23.streaming.gsvector.module.has_index_plan', return_value=True)
    def test_maintenance_while_temporarily_disabled_invalidates_prior_cache(self, plan):
        self.configure()
        self.adapter._query_chunk(self.q, 3)
        self.configure(cache_session_setup=False)
        self.adapter._mark_query_sessions_stale()
        self.configure()
        self.adapter._query_chunk(self.q, 3)
        self.assertEqual(self.setups(), 2)

    @patch('neurips23.streaming.gsvector.module.has_index_plan', return_value=True)
    def test_failed_query_closes_and_never_retries(self, plan):
        self.configure()
        original = self.conn.execute.side_effect
        def execute(statement, *args):
            if str(statement).startswith('SELECT id FROM base'):
                raise RuntimeError('simulated failure')
            return original(statement, *args)
        self.conn.execute.side_effect = execute
        with self.assertRaisesRegex(RuntimeError, 'simulated failure'):
            self.adapter._query_chunk(self.q, 3)
        self.assertEqual(self.setups(), 1)
        self.conn.close.assert_called_once()
        self.assertFalse(self.adapter._query_connections)
        self.assertIsNone(self.adapter._query_local.connection)

    @patch('neurips23.streaming.gsvector.module.has_index_plan', return_value=True)
    def test_different_connection_requires_fresh_setup(self, plan):
        self.configure()
        self.adapter._query_chunk(self.q, 3)
        replacement = self.connection()
        self.adapter._query_local.connection = replacement
        self.adapter._query_connections.append(replacement)
        self.adapter._query_chunk(self.q, 3)
        self.assertTrue(any('EXPLAIN (FORMAT JSON)' in str(c.args[0]) for c in replacement.execute.call_args_list))

    @patch('neurips23.streaming.gsvector.module.has_index_plan', return_value=True)
    def test_closed_connection_is_replaced_and_cache_is_not_reused(self, plan):
        self.configure()
        self.adapter._query_chunk(self.q, 3)
        self.conn.closed = True
        replacement = self.connection()
        with patch('neurips23.streaming.gsvector.module.psycopg.connect', return_value=replacement) as connect:
            self.adapter._query_chunk(self.q, 3)
        connect.assert_called_once()
        self.assertNotIn(self.conn, self.adapter._query_connections)
        self.assertIn(replacement, self.adapter._query_connections)
        self.assertTrue(any('EXPLAIN (FORMAT JSON)' in str(c.args[0]) for c in replacement.execute.call_args_list))

    @patch('neurips23.streaming.gsvector.module.has_index_plan', return_value=False)
    def test_failed_plan_is_never_cached(self, plan):
        self.configure()
        with self.assertRaisesRegex(RuntimeError, 'ANN index plan required'):
            self.adapter._query_chunk(self.q, 3)
        self.assertIsNone(self.adapter._query_local.session_state)
        self.conn.close.assert_called_once()


@unittest.skipUnless(os.environ.get('GSVECTOR_PG_DSN'), 'requires an isolated PostgreSQL test cluster')
class SessionCachePostgresTests(unittest.TestCase):
    def test_reuse_setup_invalidation_and_standard_sql_results(self):
        vectors = np.random.default_rng(20261031).normal(size=(256, 32)).astype('float32')
        with tempfile.TemporaryDirectory() as directory, patch.dict(os.environ, {
                'GSVECTOR_ARTIFACTS': directory, 'GSVECTOR_KEEP_SCHEMA': '0'}):
            adapter = Gsvector('euclidean', {'R': 8, 'L': 32, 'build_threads': 1,
                                             'batch_rows': 256, 'min_run_rows': 0})
            calls = []
            configure = adapter._configure_query_session
            def counted(conn, queries, k, statement):
                calls.append((conn.info.backend_pid, k))
                return configure(conn, queries, k, statement)
            adapter._configure_query_session = counted
            try:
                adapter.setup('float32', len(vectors), 32)
                adapter.insert(vectors, np.arange(len(vectors)))
                adapter.conn.execute("CREATE FUNCTION public.session_cache_prepare(regclass) RETURNS jsonb "
                                     "AS '$libdir/gsvector','gsvector_prepare_refine' LANGUAGE C STRICT")
                adapter.conn.execute('SELECT public.session_cache_prepare(%s::regclass)',
                                     (f'{adapter.schema}.ann_idx',))
                args = {'search_l': 100, 'threads': 1, 'reuse_connections': True,
                        'cache_session_setup': True, 'refine_sq8': True}
                adapter._set_query_arguments(args, reset_workload=False)
                adapter.query(vectors[:8], 10)
                reference = adapter.get_results().copy()
                self.assertEqual(len(calls), 1)
                adapter.query(vectors[:8], 10)
                np.testing.assert_array_equal(adapter.get_results(), reference)
                self.assertEqual(len(calls), 1)
                adapter._set_query_arguments({**args, 'search_l': 128}, reset_workload=False)
                adapter.query(vectors[:8], 10)
                self.assertEqual(len(calls), 2)
                adapter.query(vectors[:8], 5)
                self.assertEqual(len(calls), 3)
                old = list(adapter._query_connections)
                adapter.invalidate_query_sessions()
                self.assertTrue(all(conn.closed for conn in old))
                adapter.query(vectors[:8], 10)
                self.assertEqual(len(calls), 4)
                adapter._set_query_arguments({**args, 'cache_session_setup': False}, reset_workload=False)
                adapter.query(vectors[:8], 10)
                adapter.query(vectors[:8], 10)
                self.assertEqual(len(calls), 6)
            finally:
                adapter.done()

if __name__ == '__main__': unittest.main()
