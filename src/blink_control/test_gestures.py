"""Unit tests for the gesture recognizer. Run: python3 -m unittest src/blink_control/test_gestures.py"""
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gestures import GestureRecognizer, recognize  # noqa: E402


class GestureTests(unittest.TestCase):
    def cmds(self, times, **kw):
        return [c["command"] for c in recognize(times, **kw)]

    def test_single_blinks_do_nothing(self):
        self.assertEqual(self.cmds([1, 4, 7.5, 12]), [])

    def test_counts_map_to_commands(self):
        self.assertEqual(self.cmds([1, 1.5]), ["NEXT"])
        self.assertEqual(self.cmds([1, 1.5, 2.0]), ["PREVIOUS"])
        self.assertEqual(self.cmds([1, 1.5, 2.0, 2.5]), ["PLAY_PAUSE"])

    def test_gap_too_long_splits_burst(self):
        self.assertEqual(self.cmds([1, 2.0], max_gap=0.8), [])

    def test_too_many_blinks_ignored(self):
        self.assertEqual(self.cmds([1, 1.5, 2, 2.5, 3, 3.5]), [])

    def test_cooldown_blocks_followup(self):
        # NEXT decided at 1.5+0.8=2.3, blocked until 3.3: the pair at 2.6/3.1 is ignored
        self.assertEqual(self.cmds([1, 1.5, 2.6, 3.1]), ["NEXT"])
        self.assertEqual(self.cmds([1, 1.5, 4.0, 4.5]), ["NEXT", "NEXT"])

    def test_live_flush(self):
        rec = GestureRecognizer()
        self.assertIsNone(rec.push(10.0))
        self.assertIsNone(rec.push(10.5))
        self.assertIsNone(rec.flush(11.0))          # still waiting for a 3rd blink
        self.assertEqual(rec.flush(11.4)["command"], "NEXT")


if __name__ == "__main__":
    unittest.main()
