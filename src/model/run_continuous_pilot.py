#!/usr/bin/env python3
"""Compare EEG blink detectors on full reviewed streams held out by recording."""

import argparse
import csv
from dataclasses import dataclass
import heapq
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn

from src.eeg_preprocessing.prepare_model_data import SESSIONS, read_adc, read_csv
from src.model.run_blink_pilot import ROOT, TinyCNN, waveform_features


WINDOW = 1000
STRIDE = 100
REFRACTORY = 350
MATCH_MARGIN = 150
QUALITY_LIMIT = 0.10
RATIO = 3
MODELS = (
    "amplitude", "adaptive_amplitude", "waveform_svm", "cnn", "cnn_polarity",
    "cnn_polarity_normalized", "cnn_hard_negative", "svm_cnn_ensemble",
)
MIN_PRECISION = 0.80
MIN_RECALL = 0.20


@dataclass
class Stream:
    session: str
    centers: np.ndarray
    signal: np.ndarray
    valid: np.ndarray
    labels: np.ndarray
    blinks: np.ndarray
    closures: np.ndarray
    duration_minutes: float


def read_streams(data, raw_dir, ref, quality_limit):
    coverage = read_csv(data / "reviewed_coverage.csv")
    events = read_csv(data / "eeg_events.csv")
    streams = {}
    for interval in coverage:
        session = interval["session"]
        if session in streams:
            raise ValueError(f"Expected one reviewed interval for {session}")
        _, name, channel = SESSIONS[session]
        indices, adc = read_adc(name, channel, raw_dir, ref)
        begin, end = (int(interval[key]) for key in ("start_sample", "end_sample"))
        if not 0 <= begin < end <= indices[-1] + 1:
            raise ValueError(f"Invalid reviewed interval for {session}")
        annotations = [row for row in events if row["session"] == session]

        def bounds(label):
            return np.array([
                [int(row[f"{point}_sample"]) for point in ("onset", "peak", "offset")]
                for row in annotations if row["event_type"] == label
                and begin <= int(row["peak_sample"]) < end
            ], dtype=int).reshape(-1, 3)

        blinks, closures = bounds("blink"), bounds("sustained_closure")
        if len(blinks) == 0:
            raise ValueError(f"No reviewed blink labels in {session}")
        starts = np.arange(begin, end - WINDOW + 1, STRIDE)
        centers = starts + WINDOW // 2
        left = np.searchsorted(indices, starts)
        right = np.searchsorted(indices, starts + WINDOW)
        contiguous = right - left == WINDOW
        raw_windows = np.zeros((len(starts), WINDOW), dtype=np.int16)
        raw_windows[contiguous] = adc[left[contiguous, None] + np.arange(WINDOW)]
        clips = np.mean((raw_windows == 0) | (raw_windows >= 1018), axis=1)
        valid = contiguous & (clips <= quality_limit) & (np.ptp(raw_windows, axis=1) >= 10)
        centered = (raw_windows - np.median(raw_windows, axis=1, keepdims=True)) / 512
        signal = resample_poly(centered, 1, 10, axis=1).astype(np.float32)

        labels = np.zeros(len(centers), dtype=np.int8)
        for onset, _, offset in np.concatenate((blinks, closures)):
            labels[(starts < offset + MATCH_MARGIN) & (starts + WINDOW > onset - MATCH_MARGIN)] = -1
        # Use webcam-aligned peaks, including examples without an EEG dip match.
        for _, peak, _ in blinks:
            labels[np.abs(centers - peak) <= STRIDE] = 1
        for onset, _, offset in closures:
            labels[(centers >= onset) & (centers <= offset)] = -1
        labels[~valid] = -1
        streams[session] = Stream(session, centers, signal, valid, labels, blinks, closures,
                                  (end - begin) / 60_000)
    if len(streams) != 3:
        raise ValueError("This pilot requires three covered recordings")
    return streams


def orient(x):
    smooth = gaussian_filter1d(x, sigma=2, axis=1)
    peak = smooth[np.arange(len(x)), np.abs(smooth).argmax(axis=1)]
    return x * np.where(peak < 0, -1, 1)[:, None]


def normalize_noise(x):
    # Background variation estimate from first differences across the input window.
    scale = np.median(np.abs(np.diff(x, axis=1)), axis=1, keepdims=True)
    return np.clip(x / np.maximum(scale * 18, 0.25), -4, 4).astype(np.float32)


def shape_features(x):
    oriented = orient(x)
    relative = normalize_noise(oriented)
    smooth = gaussian_filter1d(oriented, sigma=2, axis=1)
    amplitude = smooth.max(axis=1)
    peak = smooth.argmax(axis=1)
    # Compare the neighborhoods of the dominant deflection to its height.
    positions = (peak[:, None] + np.array([-18, -10, -5, 5, 10, 18])).clip(0, x.shape[1] - 1)
    profile = smooth[np.arange(len(x))[:, None], positions] / np.maximum(amplitude[:, None], 0.03)
    return np.column_stack([waveform_features(oriented), waveform_features(relative), profile])


def train_data(streams):
    positive = np.concatenate([s.signal[s.labels == 1] for s in streams])
    negative = [s.signal[s.labels == 0] for s in streams]
    counts = [int(np.count_nonzero(s.labels == 1)) for s in streams]
    if any(n == 0 or len(background) < RATIO * n for n, background in zip(counts, negative)):
        raise ValueError("Each training session needs both classes at the selected ratio")
    return positive, negative, counts


def sampled_background(background, counts, rng):
    return np.concatenate([part[rng.choice(len(part), RATIO * count, replace=False)]
                           for part, count in zip(background, counts)])


def hard_background(background, counts, rng, model):
    selected = []
    with torch.no_grad():
        for part, count in zip(background, counts):
            pool = rng.choice(len(part), min(len(part), 8 * count), replace=False)
            signal = torch.from_numpy(normalize_noise(part[pool]))
            scores = torch.maximum(model(signal), model(-signal)).numpy()
            hard = pool[np.argsort(scores)[-count:]]
            random = rng.choice(np.setdiff1d(np.arange(len(part)), hard),
                                (RATIO - 1) * count, replace=False)
            selected.append(part[np.r_[hard, random]])
    return np.concatenate(selected)


def fit(name, train, seed=42, weights_dir=None):
    if name in ("amplitude", "adaptive_amplitude"):
        return lambda x: waveform_features(x)[:, 0]
    if name == "svm_cnn_ensemble":
        svm = fit("waveform_svm", train, seed, weights_dir)
        cnn = fit("cnn_polarity_normalized", train, seed, weights_dir)
        return lambda x: np.sqrt(svm(x) * cnn(x))
    positive, background, counts = train_data(train)
    rng = np.random.default_rng(seed)
    y = np.r_[np.ones(len(positive)), np.zeros(RATIO * len(positive))].astype(np.float32)
    if name == "waveform_svm":
        sampled = sampled_background(background, counts, rng)
        classifier = make_pipeline(StandardScaler(), SVC(C=2, probability=True, random_state=seed))
        classifier.fit(shape_features(np.concatenate((positive, sampled))), y)
        if weights_dir is not None:
            joblib.dump(classifier, weights_dir / "classifier.joblib")
        return lambda x: classifier.predict_proba(shape_features(x))[:, 1]

    torch.manual_seed(seed)
    model = TinyCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    objective = nn.BCEWithLogitsLoss()
    transform = normalize_noise if name in ("cnn_polarity_normalized", "cnn_hard_negative") else lambda x: x
    target = torch.from_numpy(y)
    for epoch in range(25):
        if name == "cnn_hard_negative" and epoch >= 8:
            sampled = hard_background(background, counts, rng, model)
        else:
            sampled = sampled_background(background, counts, rng)
        features = transform(np.concatenate((positive, sampled)))
        if name != "cnn":
            signs = rng.choice([-1., 1.], size=(len(features), 1))
            gain = np.exp(rng.normal(0, 0.18, size=(len(features), 1)))
            features = (features * signs * gain).astype(np.float32)
        tensor = torch.from_numpy(features)
        for batch in np.array_split(rng.permutation(len(y)), max(1, (len(y) + 63) // 64)):
            optimizer.zero_grad()
            objective(model(tensor[batch]), target[batch]).backward()
            optimizer.step()

    model.eval()
    if weights_dir is not None:
        torch.save(model.state_dict(), weights_dir / "model.pt")

    @torch.no_grad()
    def predict(x):
        return np.concatenate([
            model(torch.from_numpy(transform(batch))).sigmoid().numpy()
            for batch in np.array_split(x, max(1, (len(x) + 511) // 512))
        ])

    return predict


def candidate_peaks(predict, stream, name, localize=False):
    valid = np.flatnonzero(stream.valid)
    scores = predict(stream.signal[valid])
    if name == "adaptive_amplitude":
        # Estimate background from the preceding 20 seconds of available windows.
        history = np.full(len(stream.centers) + 200, np.nan)
        history[valid + 200] = scores
        rolling = np.lib.stride_tricks.sliding_window_view(history, 200)[valid]
        enough = (np.isfinite(rolling).sum(axis=1) >= 10)
        baseline = np.full(len(valid), 0.10)
        baseline[enough] = np.nanmedian(rolling[enough], axis=1)
        scores = scores / np.maximum(baseline, 0.05)
    locations = stream.centers[valid].copy()
    if localize:
        local_waveform = gaussian_filter1d(stream.signal[valid], sigma=2, axis=1)[:, 30:71]
        locations += 10 * (np.abs(local_waveform).argmax(axis=1) - 20)
    # Greedy suppression in score order gives a single candidate list for every threshold.
    position_order = np.argsort(locations, kind="stable")
    positions = locations[position_order]
    blocked = np.zeros(len(valid), dtype=bool)
    chosen = []
    for index in np.argsort(-scores, kind="stable"):
        if blocked[index]:
            continue
        center = locations[index]
        lo, hi = np.searchsorted(positions, [center - REFRACTORY, center + REFRACTORY], side="left")
        blocked[position_order[lo:hi]] = True
        chosen.append((int(center), float(scores[index])))
    return chosen


def event_counts(stream, detected):
    """Match detections one-to-one to expanded reviewed event intervals."""
    intervals = sorted((start - MATCH_MARGIN, end + MATCH_MARGIN)
                       for start, _, end in stream.blinks)
    pending = []
    cursor = 0
    tp = 0
    unmatched = []
    for center in sorted(detected):
        while cursor < len(intervals) and intervals[cursor][0] <= center:
            heapq.heappush(pending, intervals[cursor][1])
            cursor += 1
        while pending and pending[0] < center:
            heapq.heappop(pending)
        if pending:
            heapq.heappop(pending)
            tp += 1
        else:
            unmatched.append(center)
    alerts = np.asarray(unmatched, dtype=int)[:, None]
    closure_alerts = int(np.any((stream.closures[:, 0] <= alerts)
                                & (alerts <= stream.closures[:, 2]), axis=1).sum())
    return tp, len(detected) - tp, len(stream.blinks) - tp, closure_alerts


def metrics(tp, fp, fn):
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "precision": precision, "recall": recall,
        "f1": 2 * tp / max(1, 2 * tp + fp + fn),
        "f0_5": 1.25 * tp / max(1., 1.25 * tp + fp + 0.25 * fn),
    }


def calibrate(validation, criterion):
    scores = np.array([score for _, candidates in validation for _, score in candidates])
    thresholds = np.unique(np.quantile(scores, np.linspace(0.15, 1, 121)))
    points = []
    for threshold in thresholds:
        counts = np.sum([event_counts(stream, [center for center, score in candidates if score >= threshold])
                         for stream, candidates in validation], axis=0)
        values = metrics(*counts[:3])
        points.append((float(threshold), values))
    if criterion == "precision":
        qualified = [p for p in points if p[1]["precision"] >= MIN_PRECISION
                     and p[1]["recall"] >= MIN_RECALL]
        if qualified:
            return max(qualified, key=lambda p: (p[1]["recall"], p[1]["precision"]))
        return max(points, key=lambda p: p[1]["f0_5"])
    return max(points, key=lambda p: p[1]["f1"])


def run(streams, out, models, quality_limit, localize):
    results = []
    predictions = []
    for session, test in streams.items():
        train = [stream for name, stream in streams.items() if name != session]
        for name in models:
            validation = []
            for source, held_out in (train, train[::-1]):
                validation.append((held_out, candidate_peaks(fit(name, [source]), held_out, name, localize)))
            policies = {policy: calibrate(validation, policy) for policy in ("f1", "precision")}
            candidates = candidate_peaks(fit(name, train), test, name, localize)
            for policy, (threshold, validation_stats) in policies.items():
                retained = [(center, score) for center, score in candidates if score >= threshold]
                tp, fp, fn, alerts = event_counts(test, [center for center, _ in retained])
                results.append({
                    "session": session, "model": name, "policy": policy,
                    "threshold": threshold, "validation": validation_stats,
                    **metrics(tp, fp, fn), "true_positives": tp, "false_positives": fp,
                    "missed_blinks": fn, "closure_false_alarms": alerts,
                    "false_alarms_per_minute": fp / test.duration_minutes,
                })
                predictions.extend((session, name, policy, center, score) for center, score in retained)
            print(f"Completed {session} / {name}", flush=True)
    totals = {}
    for name in models:
        for policy in ("f1", "precision"):
            runs = [r for r in results if r["model"] == name and r["policy"] == policy]
            tp, fp, fn = [sum(r[key] for r in runs)
                          for key in ("true_positives", "false_positives", "missed_blinks")]
            totals[f"{name}/{policy}"] = {
                **metrics(tp, fp, fn), "true_positives": tp, "false_positives": fp, "missed_blinks": fn,
                "false_alarms_per_minute": fp / sum(s.duration_minutes for s in streams.values()),
            }
    out.mkdir(parents=True, exist_ok=True)
    metadata = {key: {"reviewed_minutes": s.duration_minutes, "reviewed_blinks": len(s.blinks),
                      "quality_pass_fraction": float(s.valid.mean()),
                      "training_positive_windows": int(np.count_nonzero(s.labels == 1)),
                      "training_background_candidates": int(np.count_nonzero(s.labels == 0))}
                for key, s in streams.items()}
    (out / "metrics.json").write_text(json.dumps({
        "max_clipped_fraction": quality_limit, "window_samples": WINDOW, "stride_samples": STRIDE,
        "suppression_samples": REFRACTORY, "event_matching_margin_samples": MATCH_MARGIN,
        "localize_peak": localize,
        "streams": metadata, "pooled": totals, "held_out": results,
    }, indent=2) + "\n")
    with (out / "detections.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("session", "model", "policy", "detection_sample", "score"))
        writer.writerows(predictions)
    print(json.dumps(totals, indent=2))


def freeze_candidate(streams, out, name, quality_limit, localize):
    """Calibrate on out-of-fold predictions and fit a candidate for a future independent test."""
    validation = []
    for session, held_out in streams.items():
        train = [stream for key, stream in streams.items() if key != session]
        validation.append((held_out, candidate_peaks(fit(name, train), held_out, name, localize)))
    out.mkdir(parents=True, exist_ok=True)
    policies = {}
    for policy in ("f1", "precision"):
        threshold, scores = calibrate(validation, policy)
        policies[policy] = {"threshold": threshold, "noor_out_of_fold_calibration": scores}
    fit(name, list(streams.values()), weights_dir=out)
    config = {
        "model": name, "calibration_sessions": list(streams), "training_seed": 42,
        "max_clipped_fraction": quality_limit, "window_samples": WINDOW,
        "stride_samples": STRIDE, "suppression_samples": REFRACTORY,
        "localize_peak": localize, "policies": policies,
    }
    (out / "detector.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/model")
    parser.add_argument("--raw-dir", type=Path, default=ROOT / "data/eeg/raw")
    parser.add_argument("--eeg-ref", default="43f284c")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/blink_continuous")
    parser.add_argument("--max-clipped-fraction", type=float, choices=(0.10, 0.25), default=QUALITY_LIMIT)
    parser.add_argument("--models", choices=MODELS, nargs="+", default=MODELS)
    parser.add_argument("--localize-peak", action="store_true",
                        help="Place each alert at a nearby EEG deflection")
    parser.add_argument("--freeze-model", choices=MODELS,
                        help="Fit one candidate on all three recordings for a future external test")
    args = parser.parse_args()
    torch.set_num_threads(1)
    streams = read_streams(args.data, args.raw_dir, args.eeg_ref, args.max_clipped_fraction)
    if args.freeze_model:
        freeze_candidate(streams, args.output, args.freeze_model, args.max_clipped_fraction, args.localize_peak)
    else:
        run(streams, args.output, args.models, args.max_clipped_fraction, args.localize_peak)


if __name__ == "__main__":
    main()
