#!/usr/bin/env python3
"""Clean reviewed event labels and prepare fixed-length EEG model inputs."""

import argparse
import csv
import io
import json
from pathlib import Path
import subprocess

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
BOUNDARIES = ("onset", "peak", "offset")
KEY = ("session", "event_id", "event_type")
WEBCAM_FIELDS = KEY + tuple(f"{point}_{unit}" for point in BOUNDARIES for unit in ("frame", "s"))
EEG_FIELDS = KEY + tuple(f"{point}_sample" for point in BOUNDARIES) + ("use_for_training",)
WINDOW_FIELDS = ("session", "subject", "event_id", "label", "start_sample", "end_sample")
CLASS_CODES = {"no_blink": 0, "blink": 1, "sustained_closure": 2}
SESSIONS = {
    "noor1": ("noor", "noor_recording1.txt", "A3"),
    "noor2": ("noor", "noor_recording2.txt", "A4"),
    "noor3": ("noor", "noor recording_3.txt", "A2"),
    "rahma1": ("rahma", "rahma_recording1.txt", "A4"),
}
WINDOW_SAMPLES = 1000
FS = 1000
MAX_RAIL_FRACTION = 0.10
EVENT_MARGIN = 150


def read_csv(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def ordered_triplet(row, unit, parse):
    fields = [f"{point}_{unit}" for point in BOUNDARIES]
    values = [parse(row[field]) for field in fields]
    if not (0 <= values[0] <= values[1] <= values[2]) or any(not np.isfinite(v) for v in values):
        raise ValueError(f"Invalid {unit} boundaries in event {row['event_id']}")
    return fields, values


def clean_labels(source_dir, output):
    """Input is one reviewed CSV per session with optional existing EEG sample indices."""
    webcam, eeg = [], []
    for session in SESSIONS:
        source = source_dir / f"{session}.csv"
        if not source.is_file():
            continue
        reviewed = [row for row in read_csv(source)
                    if row["review_status"].strip() == "accepted"
                    and row["human_review_status"].strip() == "completed"
                    and row["event_type"].strip() in ("blink", "sustained_closure")]
        reviewed.sort(key=lambda row: (float(row["onset_s"]), float(row["peak_s"]), int(row["event_id"])))
        unique = set()
        next_id = 0
        for row in reviewed:
            frame_fields, frames = ordered_triplet(row, "frame", int)
            time_fields, times = ordered_triplet(row, "s", float)
            event_type = row["event_type"].strip()
            deduplication_key = (event_type, *frames)
            if deduplication_key in unique:
                continue
            unique.add(deduplication_key)
            next_id += 1
            event = {"session": session, "event_id": next_id, "event_type": event_type}
            webcam.append({**event, **dict(zip(frame_fields, frames)), **dict(zip(time_fields, times))})
            if all(row.get(f"{point}_sample") for point in BOUNDARIES):
                fields, samples = ordered_triplet(row, "sample", int)
                eligible = row.get("use_for_training", "1")
                if eligible not in ("0", "1"):
                    raise ValueError(f"Invalid use_for_training value in {session} event {next_id}")
                eeg.append({**event, **dict(zip(fields, samples)), "use_for_training": eligible})
    if not webcam:
        raise ValueError(f"No reviewed inputs found under {source_dir}")
    write_csv(output / "webcam_events.csv", webcam, WEBCAM_FIELDS)
    write_csv(output / "eeg_events.csv", eeg, EEG_FIELDS)


def read_adc(name, channel, raw_dir, ref):
    path = raw_dir / name
    data = path.read_bytes() if path.is_file() else subprocess.check_output(
        ["git", "show", f"{ref}:data/eeg/raw/{name}"], cwd=ROOT
    )
    with io.BytesIO(data) as file:
        file.readline()
        metadata, = json.loads(file.readline()[2:]).values()
        columns = metadata["column"]
        adc_column = columns.index(channel)
        if metadata["sampling rate"] != FS or metadata["resolution"][adc_column] != 10:
            raise ValueError(f"Unexpected sampling rate or ADC resolution in {name}")
        raw = np.loadtxt(file, dtype=np.int16, usecols=(columns.index("nSeq"), adc_column), ndmin=2)
    sequence, adc = raw.T
    if np.any((sequence < 0) | (sequence > 15)) or np.any((adc < 0) | (adc > 1023)):
        raise ValueError(f"Invalid packet counter or ADC values in {name}")
    indices = np.r_[0, np.cumsum(1 + (np.diff(sequence) - 1) % 16)]
    return indices, adc


def prepare_windows(eeg_labels, reviewed_open_eyes, raw_dir, ref, output):
    annotations = read_csv(eeg_labels)
    negatives = read_csv(reviewed_open_eyes) if reviewed_open_eyes else []
    if any(row["session"] not in SESSIONS for row in [*annotations, *negatives]):
        raise ValueError("Unrecognized EEG session")
    windows, manifest = [], []
    stats = {}
    for session, (subject, name, channel) in SESSIONS.items():
        events = [row for row in annotations if row["session"] == session]
        intervals = [row for row in negatives if row["session"] == session]
        if not events:
            if intervals:
                raise ValueError(f"Reviewed intervals have no corresponding EEG annotations for {session}")
            continue
        indices, adc = read_adc(name, channel, raw_dir, ref)
        boundaries = [ordered_triplet(event, "sample", int)[1] for event in events]
        if any(left < 0 or right > indices[-1] for left, _, right in boundaries):
            raise ValueError(f"Annotated samples outside {session}")
        outcomes = {"examples": {label: 0 for label in CLASS_CODES}, "excluded_annotation": 0,
                    "overlaps_event": 0, "edge_or_gap": 0, "clipping_or_flat": 0}
        stats[session] = outcomes
        starts = set()

        def add_window(start, label, event_id=None, event_index=None):
            end = start + WINDOW_SAMPLES
            margin = EVENT_MARGIN if label == "no_blink" else 0
            if any(i != event_index and start - margin <= right and end + margin > left
                   for i, (left, _, right) in enumerate(boundaries)):
                outcomes["overlaps_event"] += 1
                return
            a, b = np.searchsorted(indices, [start, end])
            if start < 0 or end > indices[-1] + 1 or b - a != WINDOW_SAMPLES:
                outcomes["edge_or_gap"] += 1
                return
            signal = adc[a:b]
            if np.mean((signal == 0) | (signal >= 1018)) > MAX_RAIL_FRACTION or np.ptp(signal) < 10:
                outcomes["clipping_or_flat"] += 1
                return
            if start in starts:
                return
            starts.add(start)
            x = (signal.astype(np.float32) - np.float32(np.median(signal))) / np.float32(512)
            windows.append(x)
            manifest.append(dict(session=session, subject=subject, event_id=event_id, label=label,
                                 start_sample=start, end_sample=end))
            outcomes["examples"][label] += 1

        for i, (event, (onset, peak, end)) in enumerate(zip(events, boundaries)):
            label = event["event_type"]
            if label not in ("blink", "sustained_closure") or event["use_for_training"] not in ("0", "1"):
                raise ValueError(f"Invalid reviewed event in {session}")
            if event["use_for_training"] == "0" or (label == "blink" and end - onset > 800):
                outcomes["excluded_annotation"] += 1
                continue
            start = peak - WINDOW_SAMPLES // 2
            if label == "blink":
                start = max(end - WINDOW_SAMPLES + 1, min(onset, start))
            add_window(start, label, event["event_id"], i)
        for interval in intervals:
            start, end = int(interval["start_sample"]), int(interval["end_sample"])
            if not 0 <= start < end <= indices[-1] + 1:
                raise ValueError(f"Reviewed open-eye interval outside {session}")
            for index in range(start, end - WINDOW_SAMPLES + 1, WINDOW_SAMPLES):
                add_window(index, "no_blink")
    if not windows:
        raise ValueError("No EEG examples passed preparation")
    order = sorted(range(len(manifest)), key=lambda i: (manifest[i]["session"], manifest[i]["start_sample"]))
    manifest = [manifest[i] for i in order]
    write_csv(output / "window_index.csv", manifest, WINDOW_FIELDS)
    np.savez_compressed(output / "eeg_windows.npz", X=np.stack([windows[i] for i in order]),
                        y=np.array([CLASS_CODES[row["label"]] for row in manifest], dtype=np.int8))
    counts = {label: sum(row["label"] == label for row in manifest) for label in CLASS_CODES}
    summary = {"sample_rate_hz": FS, "window_samples": WINDOW_SAMPLES, "class_codes": CLASS_CODES,
               "examples": counts, "sessions": stats, "binary_blink_training_ready":
                   bool(counts["blink"] and counts["no_blink"])}
    (output / "dataset_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "data" / "model")
    commands = parser.add_subparsers(dest="command", required=True)
    clean = commands.add_parser("clean", help="Clean reviewed event CSVs")
    clean.add_argument("--source-dir", type=Path, required=True)
    windows = commands.add_parser("windows", help="Build EEG tensors from prepared sample annotations")
    windows.add_argument("--eeg-events", type=Path)
    windows.add_argument("--reviewed-open-eyes", type=Path)
    windows.add_argument("--raw-dir", type=Path, default=ROOT / "data" / "eeg" / "raw")
    windows.add_argument("--eeg-ref", default="origin/feat/eeg-preprocessing")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.command == "clean":
        clean_labels(args.source_dir, args.output)
    else:
        prepare_windows(args.eeg_events or args.output / "eeg_events.csv", args.reviewed_open_eyes,
                        args.raw_dir, args.eeg_ref, args.output)


if __name__ == "__main__":
    main()
