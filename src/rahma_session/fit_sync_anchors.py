"""Anchor-based webcam -> EEG clock map for Rahma 1 (author: Rahma Bezzaouia).

Problem
-------
Rahma 1 has no capture-timestamp log, so the automatic grid search in
src/eeg_preprocessing/synchronize_webcam.py produced only an *unvalidated*
map: 15/44 held-out blinks matched an EEG dip within 150 ms, while shifted
controls matched 6-12/44. That is barely better than chance.

Approach
--------
1. `candidates`: list the most prominent EEG dips (strong, isolated blinks)
   so they can be found by eye in the eye-camera video.
2. Manually write anchors.csv with columns `video_s,eeg_s` for 6-10 clear
   blinks spread over the whole recording (beginning, middle, end).
3. `fit`: least-squares fit of eeg_s = rate * video_s + offset on the anchors,
   leave-one-out residuals, then project every reviewed blink and check it
   against EEG dips, with time-shifted controls as a chance baseline.

The map is trustworthy only if (a) leave-one-out residuals are small
(< ~50 ms) and (b) the match rate is clearly above every shifted control.

Usage
-----
python3 src/rahma_session/fit_sync_anchors.py candidates \
    --eeg data/eeg/raw/rahma_recording1.txt --channel A4 --top 40

python3 src/rahma_session/fit_sync_anchors.py fit \
    --eeg data/eeg/raw/rahma_recording1.txt --channel A4 \
    --anchors data/synchronization/inputs/rahma1_anchors.csv \
    --labels data/derived/noor/session4/rahma1_labels_cleaned.csv \
    --out data/synchronization/rahma1_anchor
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eeg_preprocessing"))
from synchronize_webcam import detect_dips, read_opensignals  # noqa: E402  (reuse team code)

TOLERANCE_S = 0.150
SHIFTS_S = (-10, -5, -2, 2, 5, 10)


def load_dips(eeg_path, channel):
    meta, adc, original_indices, _, _ = read_opensignals(Path(eeg_path).read_bytes(), channel)
    fs = int(meta["sampling rate"])
    rows, prominence = detect_dips(adc, fs, original_indices)
    return original_indices[rows] / fs, prominence, len(adc) / fs


def cmd_candidates(args):
    times, prominence, duration = load_dips(args.eeg, args.channel)
    gaps = np.minimum(np.diff(times, prepend=-np.inf), np.diff(times, append=np.inf))
    table = pd.DataFrame({"eeg_s": times, "prominence_adc": prominence, "isolation_s": gaps})
    table = table[table["isolation_s"] > 1.0]           # isolated = easier to match by eye
    table = table.nlargest(args.top, "prominence_adc").sort_values("eeg_s")
    print(f"EEG duration {duration:.1f} s, {len(times)} dips detected; strongest isolated dips:")
    print(table.to_string(index=False, float_format=lambda v: f"{v:.3f}"))


def fit_line(video, eeg):
    rate, offset = np.polyfit(video, eeg, 1)
    return rate, offset


def match(predicted, dips):
    idx = np.searchsorted(dips, predicted).clip(1, len(dips) - 1)
    nearest = np.where(abs(predicted - dips[idx - 1]) <= abs(predicted - dips[idx]), idx - 1, idx)
    residual = dips[nearest] - predicted
    return residual, np.abs(residual) <= TOLERANCE_S


def cmd_fit(args):
    anchors = pd.read_csv(args.anchors)
    if len(anchors) < 4:
        raise SystemExit("Use at least 4 anchors spread over the recording (6-10 is better).")
    v, e = anchors["video_s"].to_numpy(float), anchors["eeg_s"].to_numpy(float)
    rate, offset = fit_line(v, e)

    loo = []
    for i in range(len(v)):
        keep = np.arange(len(v)) != i
        r, o = fit_line(v[keep], e[keep])
        loo.append(1000 * (e[i] - (r * v[i] + o)))
    loo = np.array(loo)

    dips, _, duration = load_dips(args.eeg, args.channel)
    labels = pd.read_csv(args.labels)
    blinks = labels[labels["event_type"].eq("blink")].copy()
    blinks["peak_eeg_s"] = rate * blinks["peak_s"].astype(float) + offset
    residual, ok = match(blinks["peak_eeg_s"].to_numpy(), dips)
    blinks["residual_ms"] = 1000 * residual
    blinks["dip_within_150ms"] = ok
    for key in ("onset", "offset"):
        blinks[f"{key}_eeg_s"] = rate * blinks[f"{key}_s"].astype(float) + offset

    controls = {s: int(match(blinks["peak_eeg_s"].to_numpy() + s, dips)[1].sum()) for s in SHIFTS_S}
    metrics = {
        "anchors": int(len(v)),
        "rate": float(rate), "offset_s": float(offset),
        "anchor_loo_abs_ms": {"median": float(np.median(abs(loo))), "max": float(np.max(abs(loo)))},
        "blinks_scored": int(len(blinks)),
        "matched_within_150ms": int(ok.sum()),
        "shifted_controls_matched": controls,
        "matched_timing_error_ms": {
            "median_signed": float(np.median(blinks.loc[ok, "residual_ms"])) if ok.any() else None,
            "median_abs": float(np.median(abs(blinks.loc[ok, "residual_ms"]))) if ok.any() else None,
        },
        "outside_eeg": int(((blinks["peak_eeg_s"] < 0) | (blinks["peak_eeg_s"] > duration)).sum()),
        "previous_grid_map": {"rate": 0.9527559695664691, "offset_s": 63.522155545308834,
                              "held_out_matched": "15/44"},
    }
    metrics["validated"] = bool(metrics["anchor_loo_abs_ms"]["max"] < 100
                                and ok.sum() > 1.5 * max(controls.values()))
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    blinks.to_csv(f"{out}_aligned_events.csv", index=False)
    Path(f"{out}_metrics.json").write_text(json.dumps(metrics, indent=2))
    print(json.dumps(metrics, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    c = sub.add_parser("candidates")
    c.add_argument("--eeg", required=True)
    c.add_argument("--channel", default="A4")
    c.add_argument("--top", type=int, default=40)
    f = sub.add_parser("fit")
    f.add_argument("--eeg", required=True)
    f.add_argument("--channel", default="A4")
    f.add_argument("--anchors", required=True)
    f.add_argument("--labels", required=True)
    f.add_argument("--out", required=True)
    args = parser.parse_args()
    {"candidates": cmd_candidates, "fit": cmd_fit}[args.command](args)


if __name__ == "__main__":
    main()
