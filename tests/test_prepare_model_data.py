"""Guard event cleanup and prevent unchecked intervals from becoming negatives."""

from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from src.eeg_preprocessing.prepare_model_data import (
    EEG_FIELDS, clean_labels, prepare_windows, read_csv, write_csv,
)


class PrepareModelDataTest(unittest.TestCase):
    def test_clean_sorts_and_relabels_only_reviewed_events(self):
        def event(identifier, start, **changes):
            return {
                "event_id": identifier, "event_type": "blink", "review_status": "accepted",
                "human_review_status": "completed", "reviewer": "private metadata",
                **{f"{point}_frame": start + i for i, point in enumerate(("onset", "peak", "offset"))},
                **{f"{point}_s": (start + i) / 30 for i, point in enumerate(("onset", "peak", "offset"))},
                **changes,
            }

        rows = [event(1, 90), event(7, 30), event(8, 30),
                event(10, 45, review_status="rejected"), event(11, 50, human_review_status="pending")]
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            write_csv(path / "noor1.csv", rows, tuple(rows[0]))
            clean_labels(path, path)
            cleaned = read_csv(path / "webcam_events.csv")
            self.assertEqual([(row["event_id"], row["onset_frame"]) for row in cleaned], [("1", "30"), ("2", "90")])
            self.assertNotIn("reviewer", cleaned[0])
            self.assertEqual(read_csv(path / "eeg_events.csv"), [])

    @patch("src.eeg_preprocessing.prepare_model_data.read_adc")
    def test_negative_windows_require_coverage_and_avoid_annotated_events(self, read_adc):
        read_adc.return_value = np.arange(5000), np.resize(np.arange(480, 540), 5000)
        event = dict(session="noor1", event_id=1, event_type="blink",
                     onset_sample=1550, peak_sample=1600, offset_sample=1650, use_for_training=1)
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder)
            eeg = path / "eeg_events.csv"
            reviewed = path / "reviewed.csv"
            write_csv(eeg, [event], EEG_FIELDS)
            write_csv(reviewed, [dict(session="noor1", start_sample=0, end_sample=3000)],
                      ("session", "start_sample", "end_sample"))
            prepare_windows(eeg, None, path, "unused", path)
            self.assertEqual([r["label"] for r in read_csv(path / "window_index.csv")], ["blink"])
            prepare_windows(eeg, reviewed, path, "unused", path)
            self.assertEqual([r["label"] for r in read_csv(path / "window_index.csv")],
                             ["no_blink", "blink", "no_blink"])
            with np.load(path / "eeg_windows.npz") as data:
                np.testing.assert_array_equal(data["y"], [0, 1, 0])
                self.assertEqual(data["X"].shape, (3, 1000))


if __name__ == "__main__":
    unittest.main()
