import os
import tempfile
import unittest
from unittest.mock import patch
from neurips23.streaming.gsvector.module import Gsvector


class RemovedRefineKnobsTest(unittest.TestCase):
    def test_only_retained_refinement_is_accepted(self):
        with tempfile.TemporaryDirectory() as tmp, patch.dict(os.environ, {
            'GSVECTOR_PG_DSN': 'dbname=unused', 'GSVECTOR_ARTIFACTS': tmp,
        }):
            algo = Gsvector('euclidean', {})
            try:
                algo.set_query_arguments({'search_l': 128, 'threads': 1, 'refine_sq8': True})
                for name, value in [('refine_expanded_pool', 8), ('refine_score_limit', 64),
                                    ('refine_u8_simd', True)]:
                    with self.subTest(name=name), self.assertRaises(ValueError):
                        algo.set_query_arguments({'search_l': 128, 'threads': 1,
                                                  'refine_sq8': True, name: value})
            finally:
                algo.done()

if __name__ == '__main__':
    unittest.main()
