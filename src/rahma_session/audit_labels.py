"""Quality audit for a reviewed webcam blink-label CSV (author: Rahma Bezzaouia).

Why this exists
---------------
The synchronization report marks Rahma 1 as an *unvalidated candidate*: 13
accepted labels overlap, one "blink" lasts over 71 s, and 16 blinks are longer
than 800 ms. Those labels cannot be used as ground truth for an unseen-person
test until each problem is checked against the eye-camera video.

This script does not delete or change anything. It writes:
  * <out>_flagged.csv  - every label plus an `issues` column and a suggested action
  * <out>_audit.md     - a summary table for the report / slides

Usage
-----
python3 src/rahma_session/audit_labels.py \
    data/derived/noor/session4/rahma1_labels_reviewed.csv \
    --out reports/rahma1_labels
"""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

FPS = 30.0                 # eye-camera playback rate used during review
DUPLICATE_PEAK_S = 0.10    # two peaks closer than this are almost surely one blink
LONG_BLINK_S = 0.80        # spontaneous blinks are typically ~0.1-0.4 s
CLOSURE_S = 1.50           # longer than this should be a sustained closure, not a blink
SHORT_BLINK_S = 0.05       # under 2 frames at 30 fps: likely a click error
FRAME_TOLERANCE = 1.5      # allowed mismatch (frames) between *_s and *_frame


def audit(df):
    df = df.copy()
    for col in ("onset_s", "peak_s", "offset_s"):
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["duration_s"] = df["offset_s"] - df["onset_s"]
    issues = {i: [] for i in df.index}

    def flag(index, text):
        issues[index].append(text)

    # 1. Boundary order and frame/time consistency
    for i, row in df.iterrows():
        if row[["onset_s", "peak_s", "offset_s"]].isna().any():
            flag(i, "missing_time")
            continue
        if not row["onset_s"] <= row["peak_s"] <= row["offset_s"]:
            flag(i, "onset_peak_offset_out_of_order")
        for key in ("onset", "peak", "offset"):
            frame = pd.to_numeric(row.get(f"{key}_frame"), errors="coerce")
            if pd.notna(frame) and abs(row[f"{key}_s"] * FPS - frame) > FRAME_TOLERANCE:
                flag(i, f"{key}_frame_mismatch")

    # 2. Duration checks (blinks only; closures are allowed to be long)
    blinks = df["event_type"].eq("blink")
    for i in df.index[blinks & (df["duration_s"] > CLOSURE_S)]:
        flag(i, "blink_longer_than_1.5s")
    for i in df.index[blinks & df["duration_s"].between(LONG_BLINK_S, CLOSURE_S, inclusive="right")]:
        flag(i, "long_blink_0.8-1.5s")
    for i in df.index[df["duration_s"] < SHORT_BLINK_S]:
        flag(i, "shorter_than_2_frames")

    # 3. Overlaps and near-duplicates (sorted by onset)
    order = df.sort_values("onset_s").index.to_list()
    for a, b in zip(order, order[1:]):
        if abs(df.at[b, "peak_s"] - df.at[a, "peak_s"]) < DUPLICATE_PEAK_S:
            flag(a, f"near_duplicate_of_{df.at[b, 'event_id']}")
            flag(b, f"near_duplicate_of_{df.at[a, 'event_id']}")
        elif df.at[b, "onset_s"] < df.at[a, "offset_s"]:
            flag(a, f"overlaps_{df.at[b, 'event_id']}")
            flag(b, f"overlaps_{df.at[a, 'event_id']}")

    # 4. Bookkeeping fields
    if "reviewer_name" in df:
        bad = df["reviewer_name"].fillna("").astype(str).str.fullmatch(r"[\W\d_a-z]{0,3}")
        for i in df.index[bad]:
            flag(i, "placeholder_reviewer_name")

    df["issues"] = [";".join(issues[i]) for i in df.index]
    df["suggested_action"] = [suggest(issues[i]) for i in df.index]
    return df


def suggest(found):
    text = ";".join(found)
    if "near_duplicate" in text:
        return "keep one label, delete the duplicate"
    if "longer_than_1.5s" in text:
        return "check video: relabel as sustained_closure or fix offset"
    if "overlaps" in text:
        return "check video: merge into one blink or fix boundaries"
    if "long_blink" in text:
        return "check video: tighten offset or mark as slow blink"
    if "out_of_order" in text or "mismatch" in text or "2_frames" in text:
        return "re-mark boundaries"
    if text == "placeholder_reviewer_name":
        return "fill reviewer_name only"
    return ""


def summary_markdown(df, source):
    blinks = df[df["event_type"].eq("blink")]
    serious = df["suggested_action"].ne("") & df["suggested_action"].ne("fill reviewer_name only")
    counts = {
        "Labels in file": len(df),
        "Accepted blinks": int(df["event_type"].eq("blink").sum()),
        "Other event types": int((~df["event_type"].eq("blink")).sum()),
        "Near-duplicate labels (peaks < 100 ms apart)": int(df["issues"].str.contains("near_duplicate").sum()),
        "Overlapping labels": int(df["issues"].str.contains("overlaps").sum()),
        "Blinks longer than 1.5 s": int(df["issues"].str.contains("longer_than_1.5s").sum()),
        "Blinks 0.8-1.5 s": int(df["issues"].str.contains("long_blink").sum()),
        "Boundary/frame problems": int(df["issues"].str.contains("order|mismatch|2_frames").sum()),
        "Labels needing a video check": int(serious.sum()),
        "Labels with no issue": int(df["issues"].eq("").sum()),
    }
    lines = [f"# Label audit: `{source}`", "", "| Check | Count |", "|---|---:|"]
    lines += [f"| {k} | {v} |" for k, v in counts.items()]
    d = blinks["duration_s"]
    lines += ["", f"Blink duration (s): median {d.median():.3f}, IQR {d.quantile(.25):.3f}-{d.quantile(.75):.3f}, "
              f"max {d.max():.2f}.", "",
              "Thresholds: duplicate < 100 ms between peaks; long blink > 0.8 s; "
              "closure > 1.5 s; frame tolerance 1.5 frames at 30 fps.", "",
              "## Labels to check in the video", "",
              "| event_id | onset_s | duration_s | issues | suggested action |", "|---|---:|---:|---|---|"]
    for _, r in df[serious].sort_values("onset_s").iterrows():
        lines.append(f"| {r['event_id']} | {r['onset_s']:.3f} | {r['duration_s']:.3f} | "
                     f"{r['issues']} | {r['suggested_action']} |")
    return "\n".join(lines) + "\n"


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path)
    parser.add_argument("--out", type=Path, required=True, help="output prefix, e.g. reports/rahma1_labels")
    args = parser.parse_args()
    df = audit(pd.read_csv(args.labels))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(f"{args.out}_flagged.csv", index=False)
    Path(f"{args.out}_audit.md").write_text(summary_markdown(df, args.labels.as_posix()))
    print(Path(f"{args.out}_audit.md").read_text())


if __name__ == "__main__":
    main()
