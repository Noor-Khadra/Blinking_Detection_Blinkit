import importlib.util
from pathlib import Path
import unittest
import numpy as np

spec = importlib.util.spec_from_file_location('candidates', Path(__file__).parents[1] / 'src/webcam/build_candidates.py')
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class CandidatesTest(unittest.TestCase):
    def test_blink_returns_first_reopened_frame(self):
        events, _ = module.propose(np.arange(7) * .033, np.array([.3, .3, .1, .1, .3, .3, .3]), reference=.3)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]['onset_frame'], 2)
        self.assertEqual(events[0]['offset_frame'], 4)
        self.assertEqual(events[0]['review_status'], 'unreviewed')

    def test_missing_tracking_splits_events(self):
        events, _ = module.propose(np.arange(7) * .033, np.array([.3, .1, .1, np.nan, .1, .1, .3]), reference=.3)
        self.assertEqual(len(events), 2)
        self.assertIn('reopening_not_observed', events[0]['flags'])

    def test_duplicate_times_fail(self):
        with self.assertRaises(ValueError):
            module.propose(np.array([0., 0.]), np.array([.3, .3]))


if __name__ == '__main__':
    unittest.main()
