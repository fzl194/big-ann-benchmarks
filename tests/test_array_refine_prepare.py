import os
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from benchmark.algorithms.pg_dense import PGArrayANN


class ArrayRefinePreparationTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.env=patch.dict(os.environ,{"GSVECTOR_PG_DSN":"dbname=not_connected",
                                       "GSVECTOR_ARTIFACTS":self.temp.name})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        self.temp.cleanup()

    def test_opt_in_is_boolean_and_metric_specific(self):
        for value in (None,1,"true"):
            with self.subTest(value=value),self.assertRaisesRegex(ValueError,"prepare_refine"):
                PGArrayANN("normalized",prepare_refine=value)
        for metric in ("ip","l1"):
            with self.subTest(metric=metric),self.assertRaisesRegex(ValueError,"L2.*cosine"):
                PGArrayANN(metric,prepare_refine=True)

    def test_omitted_preparation_does_not_touch_old_server(self):
        algo=PGArrayANN("normalized")
        algo.conn=MagicMock()
        with patch("benchmark.algorithms.pg_dense.PGDenseANN.fit") as fit:
            algo.fit("fixture")
        fit.assert_called_once_with("fixture")
        algo.conn.execute.assert_not_called()

    def test_preparation_follows_fit_and_is_counted_as_build_work(self):
        algo=PGArrayANN("normalized",prepare_refine=True)
        algo.schema='private"schema'
        conn=MagicMock()
        conn.execute.return_value.fetchone.return_value=({"runs":1,"u8_exact_runs":0},)
        def fit(_dataset):
            algo.conn=conn
            algo.build_times={"load_s":1,"index_s":2}
        algo._record=MagicMock()
        with patch("benchmark.algorithms.pg_dense.PGDenseANN.fit",side_effect=fit):
            algo.fit("fixture")
        self.assertEqual(algo.build_times["index_s"],2)
        self.assertGreaterEqual(algo.build_times["prepare_refine_s"],0)
        self.assertEqual(algo.refine_preparation["runs"],1)
        rendered=[call.args[0].as_string(None) for call in conn.execute.call_args_list]
        self.assertIn('"private""schema"',rendered[0])
        self.assertIn("gsvector_prepare_refine",rendered[0])
        self.assertTrue(rendered[1].startswith("SELECT "))
        self.assertEqual(conn.execute.call_args_list[1].args[1],('"private""schema"."ann_idx"',))
        algo._record.assert_called_once()


if __name__=="__main__":
    unittest.main()
