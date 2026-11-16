#!/usr/bin/env python3
"""Map reviewed webcam events to the original OpenSignals sample timeline."""

import argparse
import csv
import io
import json
from datetime import datetime
from pathlib import Path
import re
import subprocess
from zipfile import ZipFile

import numpy as np
from scipy.signal import butter, find_peaks, sosfiltfilt


ROOT = Path(__file__).resolve().parents[2]
DATA = Path("data/synchronization")
EEG = Path("data/eeg")
LABELS = Path("data/derived/noor")
BOUNDARIES = ("onset", "peak", "offset")
TOLERANCE = 0.150
SESSIONS = {
    "noor1": ("noor_recording1.txt", "noor_recording1.xlsx", "A3",
              LABELS / "session1/labels.csv", None, None, (-18.5, -16.3), (0.997, 1.003)),
    "noor2": ("noor_recording2.txt", "noor_recording2.xlsx", "A4",
              LABELS / "session2/labels.csv", 18184, "2026-10-25T19:10:58.575523",
              (-14.5, -11.0), (0.997, 1.003)),
    "noor3": ("noor recording_3.txt", "noor_recording3.xlsx", "A2",
              LABELS / "session3/labels.csv", 18966, "2026-11-11T12:12:24.987878",
              (-5.0, 0.0), (0.997, 1.003)),
    "rahma1": ("rahma_recording1.txt", "rahma_recording1.xlsx", "A4",
               LABELS / "session4/rahma1_labels_reviewed.csv", None, None,
               (-120.0, 120.0), (0.93, 1.21)),
}


def source_bytes(path, ref):
    local = ROOT / path
    if local.is_file():
        return local.read_bytes()
    result = subprocess.run(
        ["git", "show", f"{ref}:{path.as_posix()}"], cwd=ROOT, capture_output=True, check=False
    )
    if result.returncode:
        raise FileNotFoundError(f"{path} is unavailable locally and at {ref}")
    return result.stdout


def read_opensignals(data, channel):
    lines = data.splitlines()
    metadata, = json.loads(lines[1][2:]).values()
    columns = metadata["column"]
    analog = columns.index(channel)
    if metadata["resolution"][analog] != 10:
        raise ValueError(f"Expected 10-bit EEG on {channel}")
    raw = np.loadtxt(lines[3:], dtype=np.int16)
    if raw.ndim != 2 or raw.shape[1] != len(columns):
        raise ValueError("OpenSignals sample columns differ from header")
    sequence = raw[:, columns.index("nSeq")].astype(int)
    if np.any((sequence < 0) | (sequence > 15)):
        raise ValueError("Invalid OpenSignals packet counter")
    missing = (np.diff(sequence) - 1) % 16
    original_indices = np.r_[0, np.cumsum(1 + missing)]
    channels = {
        name: {"median_adc": float(np.median(raw[:, col])), "min_adc": int(raw[:, col].min()),
               "max_adc": int(raw[:, col].max())}
        for col, name in enumerate(columns) if re.fullmatch(r"A\d+", name)
    }
    return metadata, raw[:, analog], original_indices, missing, channels


def audit_workbook(data, adc, fs):
    """Audit all data rows, including ADC, independently of the legacy headers."""
    with ZipFile(io.BytesIO(data)) as workbook:
        sheet = workbook.read("xl/worksheets/sheet1.xml")
    cells = re.compile(
        rb'<c r="A\d+"><v>(\d+)</v></c><c r="B\d+"><v>([^<]+)</v></c>'
        rb'<c r="C\d+"><v>(\d+)</v></c>'
    )
    rows = np.asarray([(int(m[1]), float(m[2]), int(m[3])) for m in cells.finditer(sheet)])
    if len(rows) != len(adc) or sheet.count(b"<row ") != len(adc) + 1:
        raise ValueError("Workbook row count or layout differs from the OpenSignals recording")
    same_index = bool(np.array_equal(rows[:, 0], np.arange(len(adc))))
    same_time = bool(np.allclose(rows[:, 1], np.arange(len(adc)) / fs, rtol=0, atol=1e-9))
    same_adc = bool(np.array_equal(rows[:, 2], adc))
    return {
        "rows_checked": len(rows), "row_index_matches": same_index,
        "time_matches_nominal_row_index": same_time, "adc_matches_selected_channel": same_adc,
    }


def load_capture(session, frame_count):
    if frame_count is None:
        return None
    path = ROOT / DATA / "inputs" / f"{session}_capture_elapsed.csv"
    with path.open(newline="") as source:
        rows = list(csv.DictReader(source))
    indices = [int(row["frame_index"]) for row in rows]
    elapsed = np.array([float(row["capture_elapsed_s"]) for row in rows])
    if indices != list(range(frame_count)) or elapsed[0] != 0 or not np.all(np.diff(elapsed) > 0):
        raise ValueError(f"{path} does not contain a complete monotonic frame lookup")
    return elapsed


def read_events(path, capture):
    with (ROOT / path).open(newline="") as source:
        events = sorted(csv.DictReader(source), key=lambda row: float(row["peak_s"]))
    if not events or len({row["event_id"] for row in events}) != len(events):
        raise ValueError(f"{path} is empty or has duplicate event IDs")
    for event in events:
        frames = [int(event[f"{key}_frame"]) for key in BOUNDARIES]
        times = [float(event[f"{key}_s"]) for key in BOUNDARIES]
        if min(frames) < 0 or frames != sorted(frames) or times != sorted(times):
            raise ValueError(f"{path} has invalid event boundaries: {event['event_id']}")
        if capture is not None:
            if frames[-1] >= len(capture):
                raise ValueError(f"{path} references a frame outside the capture log")
            if np.max(np.abs(np.array(times) - np.array(frames) / 30)) > 0.039:
                raise ValueError(f"{path} frame numbers disagree with 30 fps playback times")
            times = capture[frames]
        event["sync_times"] = np.array(times)
        event["overlap"] = False
    active = [row for row in events if row["review_status"] == "accepted"]
    for index, event in enumerate(active):
        left, _, right = event["sync_times"]
        for other in active[index + 1:]:
            other_left, _, other_right = other["sync_times"]
            if max(left, other_left) < min(right, other_right):
                event["overlap"] = other["overlap"] = True
    return events


def detect_dips(adc, fs, original_indices):
    """Filter a copy and never filter across a detected discontinuity."""
    sos = butter(3, (0.5, 12), btype="bandpass", fs=fs, output="sos")
    split = np.flatnonzero(np.diff(original_indices) != 1) + 1
    candidate_rows = []
    candidate_prominence = []
    for indices in np.split(np.arange(len(adc)), split):
        if len(indices) < 2 * fs:
            continue
        filtered = sosfiltfilt(sos, adc[indices].astype(float))
        center = np.median(filtered)
        mad = 1.4826 * np.median(np.abs(filtered - center))
        peaks, info = find_peaks(-filtered, distance=0.15 * fs,
                                 height=max(20, 1.5 * mad) - center,
                                 prominence=max(40, 2.5 * mad))
        interior = (peaks >= fs // 2) & (peaks + fs // 2 < len(indices))
        candidate_rows.extend(indices[peaks[interior]])
        candidate_prominence.extend(info["prominences"][interior])
    rows = np.array(candidate_rows, dtype=int)
    if len(rows) < 3:
        raise ValueError("Too few high-prominence EEG dips to estimate a map")
    return rows, np.array(candidate_prominence)


def nearest_dips(values, dips):
    right = np.searchsorted(dips, values).clip(1, len(dips) - 1)
    return np.where(abs(values - dips[right - 1]) <= abs(values - dips[right]), right - 1, right)


def fit_clock(fit_times, dips, offsets, rates):
    """Maximize agreement on the fit subset, then refine with robust inliers."""
    step = 0.025
    grid_offsets = np.arange(offsets[0], offsets[1] + step / 2, step)
    grid_rates = np.arange(rates[0], rates[1] + 0.000125, 0.00025)
    grid = np.arange(0, dips[-1] + 1, step)
    distance = np.abs(grid - dips[nearest_dips(grid, dips)])
    evidence = np.exp(-0.5 * (distance / 0.085) ** 2)
    candidates = []
    for rate in grid_rates:
        positions = np.rint((rate * fit_times[:, None] + grid_offsets) / step).astype(int)
        inside = (0 <= positions) & (positions < len(grid))
        score = (evidence[positions.clip(0, len(grid) - 1)] * inside).sum(axis=0)
        best = int(np.argmax(score))
        candidates.append((float(score[best]), float(rate), float(grid_offsets[best])))
        # Retain alternative offsets even where the best clock rate is the same.
        alternative = np.abs(grid_offsets - grid_offsets[best]) > 0.5
        if np.any(alternative):
            runner = int(np.argmax(np.where(alternative, score, -1)))
            candidates.append((float(score[runner]), float(rate), float(grid_offsets[runner])))
    score, rate, offset = max(candidates)
    midpoint = float(np.median(fit_times))
    alternatives = [
        entry for entry in candidates
        if abs((entry[1] - rate) * midpoint + entry[2] - offset) > 0.5
        or abs(entry[1] - rate) * np.ptp(fit_times) > 0.5
    ]
    runner = max(alternatives, default=None)
    for _ in range(4):
        predicted = rate * fit_times + offset
        matched = dips[nearest_dips(predicted, dips)]
        keep = np.abs(matched - predicted) <= TOLERANCE
        if keep.sum() < 4:
            break
        unique = np.unique(matched[keep], return_index=True)[1]
        fit_x = fit_times[keep][unique]
        fit_y = matched[keep][unique]
        if len(fit_x) < 4:
            break
        slope, shift = np.polyfit(fit_x, fit_y, 1)
        if not (rates[0] <= slope <= rates[1] and offsets[0] <= shift <= offsets[1]):
            break
        rate, offset = float(slope), float(shift)
    return rate, offset, {"grid_score": score, "alternative_grid_score": runner}


def nearest_original_index(target, original_indices):
    if target < 0 or target > original_indices[-1]:
        return None
    position = np.searchsorted(original_indices, target)
    if original_indices[position] != target:
        return None
    return target


def write_table(path, rows, columns=None):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, columns or list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summary_for(rows):
    matches = [row for row in rows if row["dip_match"] in ("unique", "ambiguous")]
    residuals = np.array([float(row["residual_s"]) for row in matches])
    return {
        "total": len(rows), "within_150ms": len(matches),
        "unique_within_150ms": sum(row["dip_match"] == "unique" for row in matches),
        "outside_eeg": sum(row["dip_match"] == "outside_eeg" for row in rows),
        "median_signed_ms": float(np.median(residuals) * 1000) if len(matches) else None,
        "median_absolute_ms": float(np.median(abs(residuals)) * 1000) if len(matches) else None,
        "p95_absolute_ms": float(np.quantile(abs(residuals), 0.95) * 1000) if len(matches) else None,
    }


def synchronize(session, eeg_ref, output):
    raw_name, sheet_name, channel, labels, frames, capture_start, offsets, rates = SESSIONS[session]
    raw_path, sheet_path = EEG / "raw" / raw_name, EEG / "derived" / sheet_name
    meta, adc, indices, missing, channels = read_opensignals(source_bytes(raw_path, eeg_ref), channel)
    fs = int(meta["sampling rate"])
    workbook = audit_workbook(source_bytes(sheet_path, eeg_ref), adc, fs)
    if not all(workbook.values()):
        raise ValueError(f"Workbook timeline or ADC check failed: {workbook}")
    capture = load_capture(session, frames)
    events = read_events(labels, capture)
    rails = (adc <= 0) | (adc >= 1018)
    dip_rows, prominence = detect_dips(adc, fs, indices)
    dips = indices[dip_rows] / fs
    eligible = [event for event in events if event["review_status"] == "accepted"
                and event["human_review_status"] == "completed"
                and event["event_type"] == "blink" and not event["overlap"]]
    for number, event in enumerate(eligible):
        event["split"] = "fit" if number % 3 == 0 else "held_out"
    fit_times = np.array([event["sync_times"][1] for event in eligible if event["split"] == "fit"])
    rate, offset, search = fit_clock(fit_times, dips, offsets, rates)
    rows = []
    for event in events:
        row = {key: value for key, value in event.items() if key not in ("sync_times", "overlap", "split")}
        row.update(time_basis="capture_elapsed" if capture is not None else "video_playback",
                   alignment_status="provisional", overlaps_accepted_event=event["overlap"],
                   split=event.get("split", "excluded"))
        predicted = rate * event["sync_times"] + offset
        for key, time, eeg_time in zip(BOUNDARIES, event["sync_times"], predicted):
            row[f"{key}_sync_input_s"] = f"{time:.9f}"
            row[f"{key}_eeg_s"] = f"{eeg_time:.6f}"
            row[f"{key}_eeg_original_sample"] = nearest_original_index(round(eeg_time * fs), indices)
        row["source_duration_s"] = f"{event['sync_times'][2] - event['sync_times'][0]:.6f}"
        row["atypical_blink_duration"] = (
            event["event_type"] == "blink" and event["review_status"] == "accepted"
            and event["sync_times"][2] - event["sync_times"][0] > 0.8
        )
        peak = predicted[1]
        row.update(dip_match="not_checked", dip_original_sample=None, residual_s=None,
                   dips_within_150ms=None, dip_prominence_adc=None, raw_clip_fraction=None,
                   raw_local_adc_span=None, sequence_gaps_near_peak=None)
        if row["peak_eeg_original_sample"] is None:
            row["dip_match"] = "outside_eeg" if not 0 <= peak <= indices[-1] / fs else "missing_sample"
        else:
            left, right = np.searchsorted(indices, [(peak - 0.25) * fs, (peak + 0.25) * fs])
            row["raw_clip_fraction"] = f"{np.mean(rails[left:right]):.4f}"
            row["raw_local_adc_span"] = int(np.ptp(adc[left:right]))
            row["sequence_gaps_near_peak"] = int(np.count_nonzero(missing[left:right - 1]))
            if event["event_type"] == "blink" and event["review_status"] == "accepted":
                a, b = np.searchsorted(dips, [peak - TOLERANCE, peak + TOLERANCE])
                row["dips_within_150ms"] = int(b - a)
                row["dip_match"] = "no_dip" if a == b else "unique" if b - a == 1 else "ambiguous"
                if a != b:
                    match = a + int(np.argmin(abs(dips[a:b] - peak)))
                    row.update(dip_original_sample=int(indices[dip_rows[match]]),
                               residual_s=f"{dips[match] - peak:.6f}",
                               dip_prominence_adc=f"{prominence[match]:.3f}")
        rows.append(row)
    counts = {}
    for row in rows:
        dip = row["dip_original_sample"]
        if dip is not None:
            counts[dip] = counts.get(dip, 0) + 1
    for row in rows:
        if counts.get(row["dip_original_sample"], 0) > 1 or (
            row["overlaps_accepted_event"] and row["dip_match"] == "unique"
        ):
            row["dip_match"] = "ambiguous"
        row["visual_review_priority"] = (
            row["overlaps_accepted_event"] or row["atypical_blink_duration"]
            or row["dip_match"] in ("ambiguous", "no_dip", "missing_sample", "outside_eeg")
            or (row["raw_local_adc_span"] is not None and row["raw_local_adc_span"] < 10)
            or (row["raw_clip_fraction"] is not None and float(row["raw_clip_fraction"]) > 0.10)
        )
    held = [row for row in rows if row["split"] == "held_out"]
    held_in_bounds = [row for row in held if row["dip_match"] != "outside_eeg"]
    shifted_counts = []
    for shift in (-10, -5, -2, 2, 5, 10):
        shifted = np.array([float(row["peak_eeg_s"]) + shift for row in held_in_bounds])
        valid = shifted[(shifted >= 0) & (shifted <= indices[-1] / fs)]
        shifted_counts.append({"shift_s": shift, "n": len(valid), "within_150ms": int(
            np.count_nonzero(abs(valid - dips[nearest_dips(valid, dips)]) <= TOLERANCE)
        )})
    held_summary = summary_for(held_in_bounds)
    control_rates = [item["within_150ms"] / item["n"] for item in shifted_counts if item["n"]]
    held_rate = held_summary["unique_within_150ms"] / max(len(held_in_bounds), 1)
    # Flag weak maps conservatively. Shifted controls are descriptive, not p-values.
    alignment_status = "unvalidated_candidate" if (
        held_rate < 0.5 or held_rate < max(control_rates, default=0) + 0.15
    ) else "provisional_visual_review_required"
    for row in rows:
        row["alignment_status"] = alignment_status
        row["visual_review_priority"] |= alignment_status == "unvalidated_candidate"
    notes = {
        "session": session, "status": alignment_status,
        "raw_path": raw_path.as_posix(), "workbook_path": sheet_path.as_posix(),
        "reviewed_labels": labels.as_posix(), "channel": channel, "sampling_rate_hz": fs,
        "eeg_start_local_time": f"{meta['date']}T{meta['time']}",
        "capture_start_local_time": capture_start,
        "channels": channels, "observed_samples": len(adc), "sequence_discontinuities": int(np.count_nonzero(missing)),
        "minimum_missing_samples": int(missing.sum()), "rail_values_adc": [0, 1018],
        "raw_clipping_fraction": float(rails.mean()), "workbook": workbook,
        "clock": {"equation": "eeg_s = rate * source_s + offset_s", "rate": rate, "offset_s": offset,
                  "estimated_on": "every third non-overlapping accepted reviewed blink, sorted by peak time",
                  "grid_offsets_s": offsets, "grid_rates": rates, **search},
        "candidate_dips": len(dips), "fit": summary_for([row for row in rows if row["split"] == "fit"]),
        "held_out": summary_for(held), "held_out_in_bounds": held_summary,
        "held_out_shifted_controls": shifted_counts,
        "accepted_blinks": sum(row["review_status"] == "accepted" and row["event_type"] == "blink" for row in rows),
        "accepted_sustained_closures": sum(
            row["review_status"] == "accepted" and row["event_type"] == "sustained_closure" for row in rows
        ),
        "overlapping_accepted_events": sum(row["overlaps_accepted_event"] for row in rows),
        "atypical_blink_durations_over_800ms": sum(bool(row["atypical_blink_duration"]) for row in rows),
        "negative_examples": 0,
        "negative_example_reason": "No independently confirmed continuous-review intervals supplied.",
    }
    if session == "rahma1":
        notes["source_provenance"] = (
            "Reviewed GitHub CSV at 83db882; compatible with the 20,397-frame Rahma 1 review "
            "video. No session-matched capture log or high-confidence shared timing anchor."
        )
    if capture_start is not None:
        notes["capture_minus_eeg_start_s"] = (
            datetime.fromisoformat(capture_start) - datetime.fromisoformat(notes["eeg_start_local_time"])
        ).total_seconds()
        notes["capture_frame_count"] = len(capture)
        notes["capture_elapsed_s"] = float(capture[-1])
        notes["capture_max_frame_interval_s"] = float(np.diff(capture).max())
    diagnostics = json.dumps(notes, indent=2) + "\n"
    write_table(output / f"{session}_aligned_events.csv", rows)
    (output / f"{session}_diagnostics.json").write_text(diagnostics)
    return notes


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session", choices=SESSIONS, action="append")
    parser.add_argument("--eeg-ref", default="origin/feat/eeg-preprocessing",
                        help="Git ref providing raw/workbook inputs absent from the working tree")
    parser.add_argument("--output", type=Path, default=ROOT / DATA)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    for session in args.session or SESSIONS:
        notes = synchronize(session, args.eeg_ref, args.output)
        print(json.dumps({key: notes[key] for key in ("session", "clock", "held_out", "held_out_in_bounds")}),
              flush=True)


if __name__ == "__main__":
    main()
