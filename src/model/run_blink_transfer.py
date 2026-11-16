"""Evaluate frozen detectors across people and produce blind Rahma candidates."""

import csv
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.signal import resample_poly
import torch

from src.eeg_preprocessing.prepare_model_data import read_adc, read_csv
from src.model import run_blink_accuracy as accuracy
from src.model import run_continuous_pilot as pilot


ROOT = pilot.ROOT
DATA = ROOT / "data/model"
OUTPUT = ROOT / "reports/blink_transfer"
BOOSTED = ROOT / "reports/blink_accuracy/candidate"
ENSEMBLE = ROOT / "reports/blink_continuous/candidates/svm_cnn_ensemble"
REF = "43f284c"
WINDOW = pilot.WINDOW
MARGIN = pilot.MATCH_MARGIN


def read_config(path):
    return json.loads((path / "detector.json").read_text())


def stream_from_adc(session, file_name, channel, clipping):
    indices, adc = read_adc(file_name, channel, ROOT / "data/eeg/raw", REF)
    end = int(indices[-1]) + 1
    starts = np.arange(0, end - WINDOW + 1, pilot.STRIDE)
    left = np.searchsorted(indices, starts)
    right = np.searchsorted(indices, starts + WINDOW)
    contiguous = right - left == WINDOW
    raw = np.zeros((len(starts), WINDOW), dtype=adc.dtype)
    raw[contiguous] = adc[left[contiguous, None] + np.arange(WINDOW)]
    clip = ((raw == 0) | (raw >= 1018)).mean(axis=1)
    valid = contiguous & (clip <= clipping) & (np.ptp(raw, axis=1) >= 10)
    centered = (raw - np.median(raw, axis=1, keepdims=True)) / 512
    signal = resample_poly(centered, 1, 10, axis=1).astype(np.float32)
    empty_events = np.empty((0, 3), dtype=int)
    return pilot.Stream(session, starts + WINDOW // 2, signal, valid,
                        np.full(len(starts), -1, dtype=np.int8), empty_events, empty_events.copy(),
                        end / 60_000), end


def external_reference(clipping):
    """Use the independent eye tracker for scoring coverage and event labels."""
    stream, end = stream_from_adc("rana1", "rana_recording1.txt", "A3", clipping)
    events = read_csv(DATA / "rana1/eeg_events.csv")
    blinks = np.asarray([[int(r[f"{k}_sample"]) for k in ("onset", "peak", "offset")]
                         for r in events if r["event_type"] == "blink"], dtype=int)
    if len(blinks) != 68 or not np.all((blinks >= 0) & (blinks < end)):
        raise ValueError("External reference covers an unexpected blink set")
    coverage = np.zeros(end, dtype=bool)
    starts = stream.centers - WINDOW // 2
    labels = stream.labels.copy()
    for row in read_csv(DATA / "rana1/reviewed_coverage.csv"):
        start, stop = (int(row[k]) for k in ("start_sample", "end_sample"))
        if not 0 <= start < stop <= end:
            raise ValueError("Invalid external eye-tracking interval")
        coverage[start:stop] = True
        labels[(starts >= start + MARGIN) & (starts + WINDOW <= stop - MARGIN)] = 0
    for start, _, stop in blinks:
        coverage[max(0, start - MARGIN):min(end, stop + MARGIN + 1)] = True
        labels[(starts < stop + MARGIN) & (starts + WINDOW > start - MARGIN)] = -1
    for _, peak, _ in blinks:
        labels[np.abs(stream.centers - peak) <= pilot.STRIDE] = 1
    labels[~stream.valid] = -1
    stream.blinks = blinks
    stream.labels = labels
    return stream, coverage


def kept_alerts(alerts, threshold, coverage=None):
    above = [(sample, score) for sample, score in alerts if score >= threshold]
    if coverage is None:
        return above, 0
    kept = [(sample, score) for sample, score in above if coverage[sample]]
    return kept, len(above) - len(kept)


def statistics(stream, alerts, threshold, coverage=None):
    kept, unscored = kept_alerts(alerts, threshold, coverage)
    tp, fp, fn, _ = pilot.event_counts(stream, [t for t, _ in kept])
    minutes = stream.duration_minutes if coverage is None else float(coverage.mean()) * stream.duration_minutes
    return {
        **pilot.metrics(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn,
        "unscored_alerts_outside_reference": unscored, "scored_minutes": minutes,
        "false_alarms_per_minute": fp / minutes, "threshold": threshold,
    }


def write_alerts(path, alerts, header=("eeg_sample", "score")):
    with path.open("w", newline="") as out:
        writer = csv.writer(out, lineterminator="\n")
        writer.writerow(header)
        writer.writerows(alerts)


def frozen_noor_on_external():
    results = {}
    boosted = read_config(BOOSTED)
    stream, coverage = external_reference(boosted["max_clipped_fraction"])
    saved = joblib.load(BOOSTED / "model.joblib")
    score = accuracy.score_stream(lambda x: saved.predict_proba(accuracy.advanced_features(x))[:, 1], stream)
    for key, policy in boosted["policies"].items():
        alerts = accuracy.proposals(stream, score, policy["parameters"])
        results[f"boosted/{key}"] = statistics(stream, alerts, policy["threshold"], coverage)
        kept, _ = kept_alerts(alerts, policy["threshold"], coverage)
        write_alerts(OUTPUT / f"noor_boosted_to_rana1_{key}.csv", kept)

    ensemble = read_config(ENSEMBLE)
    stream, coverage = external_reference(ensemble["max_clipped_fraction"])
    classifier = joblib.load(ENSEMBLE / "classifier.joblib")
    cnn = pilot.TinyCNN()
    cnn.load_state_dict(torch.load(ENSEMBLE / "model.pt", weights_only=True, map_location="cpu"))
    cnn.eval()

    @torch.no_grad()
    def predict(x):
        svm = classifier.predict_proba(pilot.shape_features(x))[:, 1]
        cnn_scores = np.concatenate([
            cnn(torch.from_numpy(pilot.normalize_noise(batch))).sigmoid().numpy()
            for batch in np.array_split(x, max(1, (len(x) + 511) // 512))
        ])
        return np.sqrt(svm * cnn_scores)

    alerts = pilot.candidate_peaks(predict, stream, "svm_cnn_ensemble", ensemble["localize_peak"])
    for key, policy in ensemble["policies"].items():
        results[f"ensemble/{key}"] = statistics(stream, alerts, policy["threshold"], coverage)
        kept, _ = kept_alerts(alerts, policy["threshold"], coverage)
        write_alerts(OUTPUT / f"noor_ensemble_to_rana1_{key}.csv", kept)
    return results


def learn_policies(training, auxiliary=(), coverage=None):
    val = []
    for fit_data, blocks, _ in pilot.blocked_calibration(training):
        predictor = accuracy.fit("boost_mined", [*fit_data, *auxiliary])
        val.extend((block, accuracy.score_stream(predictor, block)) for block in blocks)
    if coverage is None:
        return {p: accuracy.choose(val, p) for p in ("f1", "precision")}
    # Use the same eye-tracker scoring mask during external-only calibration.
    def choose_masked(policy):
        options = []
        for post, config in accuracy.POST.items():
            candidates = [(block, [(t, s) for t, s in accuracy.proposals(block, scores, config) if coverage[t]])
                          for block, scores in val]
            threshold, metrics = pilot.calibrate(candidates, policy)
            options.append((post, threshold, metrics))
        if policy == "precision":
            qualified = [o for o in options if o[2]["precision"] >= pilot.MIN_PRECISION
                         and o[2]["recall"] >= pilot.MIN_RECALL]
            if qualified:
                return max(qualified, key=lambda o: (o[2]["recall"], o[2]["precision"]))
        return max(options, key=lambda o: o[2]["f1" if policy == "f1" else "f0_5"])

    return {p: choose_masked(p) for p in ("f1", "precision")}


def external_to_noor(noor):
    external, coverage = external_reference(.45)
    policies = learn_policies([external], coverage=coverage)
    predictor = accuracy.fit("boost_mined", [external])
    results = {}
    for policy, (post, threshold, validation) in policies.items():
        per_session = {}
        for stream in noor.values():
            alerts = accuracy.proposals(stream, accuracy.score_stream(predictor, stream), accuracy.POST[post])
            per_session[stream.session] = statistics(stream, alerts, threshold)
            kept, _ = kept_alerts(alerts, threshold)
            write_alerts(OUTPUT / f"rana1_to_{stream.session}_{policy}.csv", kept)
        tp, fp, fn = (sum(x[k] for x in per_session.values()) for k in ("tp", "fp", "fn"))
        mins = sum(x["scored_minutes"] for x in per_session.values())
        results[policy] = {
            **pilot.metrics(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn,
            "false_alarms_per_minute": fp / mins, "postprocessing": post, "threshold": threshold,
            "training_only_calibration": validation, "per_recording": per_session,
        }
    return results


def blind_rahma(noor):
    # Rahma's EEG is used only for inference because its labels are provisionally aligned.
    train = [noor["noor1"], noor["noor2"]]
    extra = [pilot.read_auxiliary_windows(DATA / "rana1", 1.)]
    policies = learn_policies(train, extra)
    folder = OUTPUT / "noor12_rana1_to_rahma"
    folder.mkdir(exist_ok=True)
    predictor = accuracy.fit("boost_mined", [*train, *extra], weights_path=folder / "model.joblib")
    stream, _ = stream_from_adc("rahma1", "rahma_recording1.txt", "A4", .45)
    scores = accuracy.score_stream(predictor, stream)
    out = {}
    for policy, (post, threshold, validation) in policies.items():
        proposals = accuracy.proposals(stream, scores, accuracy.POST[post])
        alerts, _ = kept_alerts(proposals, threshold)
        write_alerts(folder / f"{policy}_alerts.csv",
                     ((sample, f"{sample/1000:.3f}", score) for sample, score in sorted(alerts)),
                     ("eeg_sample", "eeg_elapsed_seconds", "score"))
        out[policy] = {
            "unverified_alerts": len(alerts), "postprocessing": post, "threshold": threshold,
            "noor_calibration_only": validation,
        }
    cfg = {"training_recordings": ["noor1", "noor2", "rana1"], "inference_recording": "rahma1",
           "labels_used_for_rahma": False, "rahma_scored": False, "policies": out}
    (folder / "detector.json").write_text(json.dumps(cfg, indent=2) + "\n")
    return cfg


def main():
    torch.set_num_threads(1)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    noor = pilot.read_streams(DATA, ROOT / "data/eeg/raw", REF, .45)
    noor = {k: accuracy.mark_closure_background(s) for k, s in noor.items()}
    result = {"noor123_to_rana1": frozen_noor_on_external()}
    print(json.dumps(result, indent=2), flush=True)
    result["rana1_to_noor123"] = external_to_noor(noor)
    print("Completed reverse transfer.", flush=True)
    result["noor12_rana1_to_rahma1"] = blind_rahma(noor)
    (OUTPUT / "metrics.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
