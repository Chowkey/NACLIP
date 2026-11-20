"""Check refinement defaults without loading GPU dependencies."""
import unittest

from tools.run_baseline import resolve_pamr_steps


class PAMRProtocolTests(unittest.TestCase):
    def test_existing_defaults_are_preserved(self):
        for condition in ('fixed', 'guided', 'none'):
            with self.subTest(condition=condition):
                self.assertEqual(resolve_pamr_steps(condition, 'auto'), 0)
        self.assertEqual(resolve_pamr_steps('upstream', 'auto'), 10)

    def test_explicit_refinement_is_independent_of_spatial_bias(self):
        for condition in ('fixed', 'guided', 'none', 'upstream'):
            with self.subTest(condition=condition):
                self.assertEqual(resolve_pamr_steps(condition, 'on'), 10)
                self.assertEqual(resolve_pamr_steps(condition, 'off'), 0)


if __name__ == '__main__':
    unittest.main()
