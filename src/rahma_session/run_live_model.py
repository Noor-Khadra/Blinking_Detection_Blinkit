"""Run the team's live detector (src/live_blink, cnn_01 ensemble, threshold 0.8)
over a full OpenSignals recording and save every alert (author: Rahma Bezzaouia).

This is the same code path as the live app, fed from a file instead of Bluetooth.

python3 src/rahma_session/run_live_model.py data/eeg/raw/rahma_recording1.txt \
    --channel A4 --out reports/rahma1_cnn01_alerts.csv
"""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "live_blink"))
from engine import Predictor, StreamDetector  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("eeg")
    p.add_argument("--channel", default="A4")
    p.add_argument("--out", required=True)
    a = p.parse_args()
    lines = Path(a.eeg).read_text().splitlines()
    meta = list(json.loads(lines[1][2:]).values())[0]
    cols = meta["column"]
    raw = np.loadtxt(lines[3:], dtype=np.int32)
    rows = np.c_[raw[:, cols.index("nSeq")], raw[:, cols.index(a.channel)]]
    det = StreamDetector(Predictor())
    alerts = []
    for chunk in np.array_split(rows, max(1, len(rows) // 50)):
        alerts += det.push(chunk)[2]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", newline="") as f:
        w = csv.DictWriter(f, ["peak_s", "score", "emitted_s", "algorithm_delay_s"])
        w.writeheader()
        w.writerows(alerts)
    print(f"{len(rows)/1000/60:.1f} min of EEG, {len(alerts)} alerts, packet gaps {det.gaps}")


if __name__ == "__main__":
    main()
