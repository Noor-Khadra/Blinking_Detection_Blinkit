#!/usr/bin/env python3
"""Run a recording-held-out blink classification pilot."""

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.signal import resample_poly
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import torch
from torch import nn


ROOT = Path(__file__).resolve().parents[2]
MODELS = {"amplitude": (0,), "logistic": (0,), "tiny_cnn": (0, 1, 2)}
LABELS = {"no_blink": 0, "blink": 1, "sustained_closure": 2}


def read_dataset(folder):
    with np.load(folder / "eeg_windows.npz", allow_pickle=False) as dataset:
        x, y = dataset["X"], dataset["y"]
    with (folder / "window_index.csv").open(newline="") as file:
        rows = list(csv.DictReader(file))
    if x.shape != (len(rows), 1000) or not np.isfinite(x).all():
        raise ValueError("Expected finite one-second EEG windows at 1,000 Hz")
    if not np.array_equal(y, [LABELS[row["label"]] for row in rows]):
        raise ValueError("Window index and class targets differ")
    sessions = np.array([row["session"] for row in rows])
    if len(set(sessions)) != 3 or len({row["subject"] for row in rows}) != 1:
        raise ValueError("This protocol expects three recordings from one person")
    return resample_poly(x, up=1, down=10, axis=1).astype(np.float32), y, sessions, rows


def sample_training(pool, y, sessions, ratio, seed):
    """Keep every blink and sample background independently within each session."""
    rng = np.random.default_rng(seed)
    selections = []
    for session in np.unique(sessions[pool]):
        partition = pool[sessions[pool] == session]
        blinks = partition[y[partition] == 1]
        background = partition[y[partition] == 0]
        if len(blinks) == 0 or len(background) == 0:
            raise ValueError(f"{session} needs both classes for training")
        sampled = rng.choice(background, size=min(len(background), ratio * len(blinks)), replace=False)
        selections.append(np.r_[blinks, sampled])
    return np.sort(np.concatenate(selections))


def waveform_features(x):
    smooth = gaussian_filter1d(x, sigma=2, axis=1)
    amplitude = np.abs(smooth).max(axis=1)
    slope = np.abs(np.diff(smooth, axis=1)).max(axis=1)
    return np.column_stack([
        amplitude, np.ptp(smooth, axis=1), np.sqrt(np.mean(smooth**2, axis=1)),
        np.mean(np.abs(smooth) >= amplitude[:, None] / 2, axis=1),
        slope, smooth.max(axis=1), -smooth.min(axis=1),
    ])


class TinyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(1, 8, kernel_size=11, padding=5),
            nn.ReLU(),
            nn.Conv1d(8, 16, kernel_size=9, padding=4),
            nn.ReLU(),
        )
        self.head = nn.Linear(32, 1)

    def forward(self, signal):
        features = self.features(signal[:, None, :])
        pooled = torch.cat([features.amax(dim=2), features.mean(dim=2)], dim=1)
        return self.head(pooled).squeeze(1)


def fit_predict(name, x, features, y, train, predict, seed):
    if name == "amplitude":
        return features[predict, 0]
    if name == "logistic":
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        model.fit(features[train], y[train])
        return model.predict_proba(features[predict])[:, 1]

    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    model = TinyCNN()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.003)
    signal = torch.from_numpy(x)
    labels = torch.from_numpy(y.astype(np.float32))
    loss = nn.BCEWithLogitsLoss()
    for _ in range(30):
        for batch in np.array_split(rng.permutation(train), max(1, (len(train) + 31) // 32)):
            optimizer.zero_grad()
            loss(model(signal[batch]), labels[batch]).backward()
            optimizer.step()
    model.eval()
    with torch.no_grad():
        return model(signal[predict]).sigmoid().numpy()


def choose_threshold(y, scores):
    precision, recall, thresholds = precision_recall_curve(y, scores)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    # Use the highest threshold in a tie to avoid extra background alerts.
    return float(thresholds[np.flatnonzero(f1 == f1.max())[-1]])


def measure(y, scores, threshold):
    predicted = scores >= threshold
    blink, background, closure = (y == label for label in (1, 0, 2))
    tp, fp, fn = (int(np.count_nonzero(mask)) for mask in (
        blink & predicted, background & predicted, blink & ~predicted,
    ))
    use = ~closure
    return {
        "average_precision": float(average_precision_score(y[use], scores[use])),
        "precision": tp / max(tp + fp, 1), "recall": tp / (tp + fn),
        "f1": 2 * tp / max(2 * tp + fp + fn, 1),
        "true_positives": tp, "false_positives": fp, "missed_blinks": fn,
        "background_windows": int(background.sum()), "closure_windows": int(closure.sum()),
        "closures_flagged_as_blinks": int(np.count_nonzero(closure & predicted)),
    }


def run(data, out, ratio):
    x, y, sessions, rows = read_dataset(data)
    features = waveform_features(x)
    binary = np.flatnonzero(y != 2)
    folds, metrics, predictions = {}, [], []
    for held_out in np.unique(sessions):
        pool = binary[sessions[binary] != held_out]
        test = np.flatnonzero(sessions == held_out)
        training = sample_training(pool, y, sessions, ratio, seed=42)
        inner = []
        for validation_session in np.unique(sessions[pool]):
            val = pool[sessions[pool] == validation_session]
            inner_train = sample_training(pool[sessions[pool] != validation_session], y, sessions, ratio, seed=42)
            inner.append((validation_session, inner_train, val))
        folds[held_out] = {
            "training_sampled": training.tolist(), "test_all": test.tolist(),
            "threshold_folds": [
                {"validation_session": session, "training_sampled": train.tolist(), "validation_all": val.tolist()}
                for session, train, val in inner
            ],
        }
        print(f"Held out {held_out}: {len(training)} sampled training windows, {len(test)} test windows", flush=True)
        for name, seeds in MODELS.items():
            for seed in seeds:
                val_y, val_scores = [], []
                for _, train, val in inner:
                    val_y.extend(y[val])
                    val_scores.extend(fit_predict(name, x, features, y, train, val, seed))
                threshold = choose_threshold(np.array(val_y), np.array(val_scores))
                scores = fit_predict(name, x, features, y, training, test, seed)
                stats = measure(y[test], scores, threshold)
                metrics.append(dict(test_session=held_out, model=name, seed=seed, threshold=threshold, **stats))
                for index, score in zip(test, scores):
                    predictions.append(dict(test_session=held_out, model=name, seed=seed,
                                            window_row=int(index), truth=rows[index]["label"],
                                            score=float(score), predicted_blink=int(score >= threshold)))

    data_hash = {
        filename: hashlib.sha256((data / filename).read_bytes()).hexdigest()
        for filename in ("window_index.csv", "eeg_windows.npz")
    }
    averaged = {}
    for name in MODELS:
        runs = [run for run in metrics if run["model"] == name]
        averaged[name] = {key: float(np.mean([run[key] for run in runs]))
                          for key in ("average_precision", "precision", "recall", "f1")}
    out.mkdir(parents=True, exist_ok=True)
    for name, content in (
        ("splits", {"dataset_sha256": data_hash, "target_background_per_blink": ratio, "folds": folds}),
        ("metrics", {"dataset_sha256": data_hash, "held_out_runs": metrics, "mean_across_runs": averaged}),
    ):
        (out / f"{name}.json").write_text(json.dumps(content, indent=2) + "\n", encoding="utf-8")
    with (out / "predictions.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(predictions[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(predictions)
    print(json.dumps(averaged, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data/model")
    parser.add_argument("--output", type=Path, default=ROOT / "reports/blink_training")
    parser.add_argument("--negative-ratio", type=int, choices=(2, 3), default=3)
    args = parser.parse_args()
    torch.set_num_threads(1)
    run(args.data, args.output, args.negative_ratio)


if __name__ == "__main__":
    main()
