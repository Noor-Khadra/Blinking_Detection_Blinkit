"""Compare further blink detectors with blocked calibration and recording holdout."""

import argparse
import csv
from dataclasses import replace
import json
from pathlib import Path

import joblib
import numpy as np
from scipy.ndimage import gaussian_filter1d
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import torch
from torch import nn

from src.model import run_continuous_pilot as pilot


MODELS = ("prior", "shape_svm", "boost", "boost_mined", "center_cnn", "fusion")
POST = {
    "center": (0, 350, 0),
    "peak200": (20, 350, 0),
    "peak120": (12, 350, 0),
    "peak120_short": (12, 240, 0),
    "peak120_long": (12, 500, 0),
    "smooth_peak160": (16, 350, 0.8),
}


def mark_closure_background(stream):
    """Mark confirmed sustained-closure windows for targeted negative sampling."""
    labels = stream.labels.copy()
    for onset, _, offset in stream.closures:
        use = stream.valid & (labels == -1) & (stream.centers >= onset) & (stream.centers <= offset)
        for lo, _, hi in stream.blinks:
            use &= (stream.centers + 500 <= lo - 150) | (stream.centers - 500 >= hi + 150)
        labels[use] = 2
    return replace(stream, labels=labels)


def advanced_features(x):
    smoothed = gaussian_filter1d(x, 2, axis=1)
    amplitude = np.maximum(np.max(np.abs(smoothed), axis=1), .03)
    central = smoothed[:, 34:66]
    edge = np.r_[0:30, 70:100]
    fft = np.abs(np.fft.rfft(x * np.hanning(100), axis=1)) ** 2
    total = fft[:, 1:].sum(axis=1) + 1e-8
    central_peak = np.max(np.abs(central), axis=1)
    sign = np.sign(central[np.arange(len(x)), np.abs(central).argmax(axis=1)])
    core = central * sign[:, None]
    pre = smoothed[:, 15:35] * sign[:, None]
    post = smoothed[:, 65:85] * sign[:, None]
    local_noise = np.median(np.abs(np.diff(x[:, edge], axis=1)), axis=1) + .02
    return np.column_stack([
        pilot.shape_features(x), central_peak / amplitude, central_peak / local_noise,
        np.sqrt(np.mean(core**2, axis=1)) / amplitude,
        np.mean(core > central_peak[:, None] * .55, axis=1),
        np.mean(pre, axis=1) / amplitude, np.mean(post, axis=1) / amplitude,
        np.max(np.abs(smoothed[:, edge]), axis=1) / amplitude,
        *(fft[:, low:high].sum(axis=1) / total for low, high in ((1, 4), (4, 9), (9, 16), (16, 32))),
    ])


class CenterCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv1d(1, 12, 9, padding=4), nn.SiLU(),
            nn.Conv1d(12, 16, 9, padding=8, dilation=2), nn.SiLU(),
            nn.Conv1d(16, 20, 7, padding=12, dilation=4), nn.SiLU(),
        )
        self.head = nn.Sequential(nn.Linear(60, 20), nn.SiLU(), nn.Dropout(.12), nn.Linear(20, 1))

    def forward(self, signal):
        features = self.body(signal[:, None, :])
        crop = features[:, :, 34:66]
        return self.head(torch.cat((crop.amax(-1), crop.mean(-1), features.amax(-1)), dim=1)).squeeze(1)


def sample_rows(train, rng, scout=None):
    positive = np.concatenate([s.signal[s.labels == 1] for s in train])
    background = []
    for stream in train:
        count = int(np.sum(stream.labels == 1))
        available = np.flatnonzero(stream.labels == 0)
        amount = min(len(available), count * 3)
        selected = rng.choice(available, amount, replace=False)
        if scout is not None and amount:
            pool = rng.choice(available, min(len(available), max(amount, 8 * count)), replace=False)
            hard_count = min(amount, max(1, count // 2))
            hard = pool[np.argsort(scout(stream.signal[pool]))[-hard_count:]]
            closure = np.flatnonzero(stream.labels == 2)
            closure_count = min(len(closure), count // 3, amount - len(hard))
            closure = rng.choice(closure, closure_count, replace=False)
            random_pool = np.setdiff1d(available, hard)
            random = rng.choice(random_pool, amount - len(hard) - len(closure), replace=False)
            selected = np.r_[hard, closure, random]
        background.append(stream.signal[selected])
    negative = np.concatenate(background)
    return np.concatenate((positive, negative)), np.r_[np.ones(len(positive)), np.zeros(len(negative))]


def fit(name, train, seed=42, weights_path=None):
    if name == "prior":
        return pilot.fit("svm_cnn_ensemble", train, seed)
    if name == "fusion":
        a, b = (fit(key, train, seed) for key in ("boost_mined", "center_cnn"))
        return lambda x: np.sqrt(a(x) * b(x))
    rng = np.random.default_rng(seed)
    if name != "center_cnn":
        if name == "shape_svm":
            classifier = make_pipeline(StandardScaler(), SVC(C=2, probability=True, random_state=seed))
        else:
            classifier = HistGradientBoostingClassifier(max_iter=100, max_leaf_nodes=9,
                                                        min_samples_leaf=18, l2_regularization=2,
                                                        random_state=seed)
        scout = fit("shape_svm", train, seed + 1) if name == "boost_mined" else None
        x, y = sample_rows(train, rng, scout)
        classifier.fit(advanced_features(x), y)
        if weights_path is not None:
            joblib.dump(classifier, weights_path)
        return lambda x: classifier.predict_proba(advanced_features(x))[:, 1]
    torch.manual_seed(seed)
    model = CenterCNN()
    opt = torch.optim.AdamW(model.parameters(), lr=.002, weight_decay=.01)
    loss = nn.BCEWithLogitsLoss()
    for _ in range(24):
        x, y = sample_rows(train, rng)
        x = pilot.normalize_noise(x) * rng.choice([-1, 1], (len(x), 1))
        x *= np.exp(rng.normal(0, .15, (len(x), 1)))
        x += rng.normal(0, .015, x.shape)
        x, y = torch.from_numpy(x.astype(np.float32)), torch.from_numpy(y.astype(np.float32))
        for batch in np.array_split(rng.permutation(len(y)), (len(y)+63)//64):
            opt.zero_grad()
            loss(model(x[batch]), y[batch]).backward()
            opt.step()
    model.eval()

    @torch.no_grad()
    def predict(x):
        return np.concatenate([model(torch.from_numpy(pilot.normalize_noise(batch))).sigmoid().numpy()
                               for batch in np.array_split(x, max(1, (len(x)+511)//512))])

    return predict


def proposals(stream, scores, settings):
    width, refractory, smoothing = settings
    indices = np.flatnonzero(stream.valid)
    if smoothing:
        # Smooth along time and exclude rejected windows from the average.
        numerator = gaussian_filter1d(np.where(stream.valid, scores, 0.), smoothing)
        denominator = gaussian_filter1d(stream.valid.astype(float), smoothing)
        scores = numerator / np.maximum(denominator, .01)
    positions = stream.centers[indices].copy()
    if width:
        wave = gaussian_filter1d(stream.signal[indices], 2, axis=1)[:, 50-width:51+width]
        positions += 10 * (np.abs(wave).argmax(axis=1) - width)
    ordered = np.argsort(positions, kind="stable")
    sorted_times = positions[ordered]
    blocked = np.zeros(len(indices), dtype=bool)
    picks = []
    for j in np.argsort(-scores[indices], kind="stable"):
        if blocked[j]:
            continue
        point = positions[j]
        lo, hi = np.searchsorted(sorted_times, [point - refractory, point + refractory])
        blocked[ordered[lo:hi]] = True
        picks.append((int(point), float(scores[indices[j]])))
    return picks


def score_stream(predict, stream):
    scores = np.zeros(len(stream.valid))
    if stream.valid.any():
        scores[stream.valid] = predict(stream.signal[stream.valid])
    return scores


def choose(validation, objective):
    options = []
    for name, settings in POST.items():
        events = [(s, proposals(s, scores, settings)) for s, scores in validation]
        threshold, stats = pilot.calibrate(events, objective)
        options.append((name, threshold, stats))
    if objective == "precision":
        good = [x for x in options if x[2]["precision"] >= pilot.MIN_PRECISION
                and x[2]["recall"] >= pilot.MIN_RECALL]
        if good:
            return max(good, key=lambda x: (x[2]["recall"], x[2]["precision"]))
        return max(options, key=lambda x: x[2]["f0_5"])
    return max(options, key=lambda x: x[2]["f1"])


def run(streams, models, out, quality, seeds):
    rows, predictions = [], []
    for held_out, test in streams.items():
        fitting = [s for name, s in streams.items() if name != held_out]
        folds = list(pilot.blocked_calibration(fitting))
        for model in models:
            for seed in seeds:
                validation = []
                for training, holdouts, _ in folds:
                    predict = fit(model, training, seed)
                    validation.extend((s, score_stream(predict, s)) for s in holdouts)
                policies = {p: choose(validation, p) for p in ("f1", "precision")}
                test_score = score_stream(fit(model, fitting, seed), test)
                for policy, (post, threshold, cal) in policies.items():
                    alerts = [(sample, score) for sample, score in proposals(test, test_score, POST[post])
                              if score >= threshold]
                    tp, fp, fn, fp_closure = pilot.event_counts(test, [t for t, _ in alerts])
                    rows.append({
                        "test": held_out, "model": model, "seed": seed, "policy": policy,
                        "postprocessing": post, "threshold": threshold, "validation": cal,
                        **pilot.metrics(tp, fp, fn), "tp": tp, "fp": fp, "fn": fn,
                        "fp_during_closure": fp_closure, "false_alarms_per_minute": fp / test.duration_minutes,
                    })
                    predictions.extend((held_out, model, seed, policy, t, s) for t, s in alerts)
            print(f"Completed {held_out} / {model}", flush=True)
    total_minutes = sum(s.duration_minutes for s in streams.values())
    pooled = {}
    for model in models:
        for seed in seeds:
            for policy in ("f1", "precision"):
                sub = [x for x in rows if (x["model"], x["seed"], x["policy"]) == (model, seed, policy)]
                tp, fp, fn = [sum(x[m] for x in sub) for m in ("tp", "fp", "fn")]
                pooled[f"{model}/{seed}/{policy}"] = {**pilot.metrics(tp, fp, fn), "tp":tp, "fp":fp, "fn":fn,
                                                       "false_alarms_per_minute":fp/total_minutes}
    out.mkdir(parents=True, exist_ok=True)
    (out / "metrics.json").write_text(json.dumps({"quality":quality, "settings":POST, "pooled":pooled,
                                                   "held_out":rows}, indent=2) + "\n")
    with (out / "alerts.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("test","model","seed","policy","detection_sample","score"))
        writer.writerows(predictions)
    print(json.dumps(pooled, indent=2))


def freeze(streams, model, seed, quality, out):
    validation = []
    for training, holdouts, _ in pilot.blocked_calibration(streams.values()):
        predict = fit(model, training, seed)
        validation.extend((s, score_stream(predict, s)) for s in holdouts)
    policies = {}
    for name in ("f1", "precision"):
        post, threshold, stats = choose(validation, name)
        policies[name] = {
            "postprocessing": post, "parameters": POST[post], "threshold": threshold,
            "noor_calibration_only": stats,
        }
    out.mkdir(parents=True, exist_ok=True)
    fit(model, list(streams.values()), seed, out / "model.joblib")
    config = {"model": model, "seed": seed, "max_clipped_fraction": quality,
              "training_recordings": list(streams), "input_hz": 100, "window_samples": 100,
              "stride_ms": pilot.STRIDE, "policies": policies}
    (out / "detector.json").write_text(json.dumps(config, indent=2) + "\n")
    print(json.dumps(config, indent=2))

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", nargs="+", choices=MODELS, default=MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--quality", type=float, default=.25)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--freeze", choices=("shape_svm", "boost", "boost_mined"))
    args = parser.parse_args()
    torch.set_num_threads(1)
    streams = pilot.read_streams(pilot.ROOT / "data/model", pilot.ROOT / "data/eeg/raw", "43f284c", args.quality)
    streams = {key: mark_closure_background(s) for key, s in streams.items()}
    if args.freeze:
        freeze(streams, args.freeze, args.seeds[0], args.quality, args.output)
    else:
        run(streams, args.models, args.output, args.quality, args.seeds)


if __name__ == "__main__":
    main()
