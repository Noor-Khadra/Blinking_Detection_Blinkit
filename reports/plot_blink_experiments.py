"""Build the BlinkIt experiment charts and comparison tables from saved run outputs."""

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
WINDOW = ROOT / "reports/blink_training"
CONTINUOUS = ROOT / "reports/blink_continuous"
OUTPUT = ROOT / "reports/blink_experiments"
CASES = ("clip10", "clip25", "hard10", "localized25", "localized_hybrid25")
ENSEMBLE = "svm_cnn_ensemble"
BLUE = "#264D82"
TEAL = "#087F81"
GRAY = "#CFD8E3"
INK = "#172A3B"
MUTED = "#50677A"
AMBER = "#BB602A"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path):
    with path.open(newline="", encoding="utf-8") as source:
        return list(csv.DictReader(source))


def write_csv(path, rows):
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def summarize_continuous():
    summary = []
    for case in CASES:
        results = read_json(CONTINUOUS / case / "metrics.json")
        for key, pooled in results["pooled"].items():
            model, policy = key.split("/")
            rows = [r for r in results["held_out"] if r["model"] == model and r["policy"] == policy]
            summary.append({
                "experiment": case, "model": model, "policy": policy,
                "max_clipped_fraction": results["max_clipped_fraction"],
                "localized_peak": case.startswith("localized"),
                **pooled, "min_recording_recall": min(r["recall"] for r in rows),
                "closure_false_alarms": sum(r["closure_false_alarms"] for r in rows),
            })
    return summary


def audit_alerts():
    annotations = read_csv(ROOT / "data/model/eeg_events.csv")
    detections = read_csv(CONTINUOUS / "localized_hybrid25/detections.csv")
    results = read_json(CONTINUOUS / "localized_hybrid25/metrics.json")
    rows = []
    for result in results["held_out"]:
        if result["model"] != ENSEMBLE or result["policy"] != "precision":
            continue
        session = result["session"]
        events = [row for row in annotations if row["session"] == session]

        def bounds(event_type):
            return sorted((int(row["onset_sample"]), int(row["offset_sample"]))
                          for row in events if row["event_type"] == event_type)

        blinks, closures = bounds("blink"), bounds("sustained_closure")
        remaining = blinks.copy()
        alerts = sorted(int(row["detection_sample"]) for row in detections
                        if row["session"] == session and row["model"] == ENSEMBLE
                        and row["policy"] == "precision")
        counts = dict(true_positive=0, extra_alert_for_blink=0,
                      near_blink_outside_tolerance=0, inside_sustained_closure=0,
                      away_from_labeled_events=0)
        for alert in alerts:
            matches = [i for i, (left, right) in enumerate(remaining) if left - 150 <= alert <= right + 150]
            if matches:
                remaining.pop(min(matches, key=lambda i: remaining[i][1]))
                counts["true_positive"] += 1
            elif any(left <= alert <= right for left, right in closures):
                counts["inside_sustained_closure"] += 1
            elif any(left - 150 <= alert <= right + 150 for left, right in blinks):
                counts["extra_alert_for_blink"] += 1
            elif any(left - 500 <= alert <= right + 500 for left, right in blinks):
                counts["near_blink_outside_tolerance"] += 1
            else:
                counts["away_from_labeled_events"] += 1
        if (counts["true_positive"] != result["true_positives"]
                or len(alerts) - counts["true_positive"] != result["false_positives"]
                or len(remaining) != result["missed_blinks"]
                or counts["inside_sustained_closure"] != result["closure_false_alarms"]):
            raise ValueError(f"False-alarm audit differs from saved metrics for {session}")
        rows.append({"session": session, "reviewed_blinks": len(blinks), "missed_blinks": len(remaining), **counts})
    return rows


def window_chart():
    names = ("amplitude", "logistic", "tiny_cnn")
    labels = ("Amplitude", "Logistic regression", "Small 1D CNN")
    runs = [read_json(path)["mean_across_runs"] for path in (
        WINDOW / "ratio2/metrics.json", WINDOW / "metrics.json",
    )]
    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0), sharey=True)
    fig.suptitle("BlinkIt window classification", x=0.09, y=0.96, ha="left", fontsize=19, weight="bold")
    fig.text(0.09, 0.865, "Means across three held-out Noor recordings. Each model uses the same test windows.",
             color=MUTED, fontsize=10)
    y = np.arange(len(names))
    for axis, metric, title in zip(axes, ("average_precision", "f1"), ("Average precision", "Blink F1")):
        for index, (source, ratio, color) in enumerate(zip(runs, ("2:1", "3:1"), (GRAY, BLUE))):
            values = [source[name][metric] for name in names]
            bars = axis.barh(y + (index - 0.5) * 0.30, values, height=0.26, label=ratio, color=color)
            axis.bar_label(bars, labels=[f"{value:.2f}" for value in values],
                           padding=4, color=INK, fontsize=9)
        axis.set_title(title, loc="left", fontsize=12, weight="bold")
        axis.set_xlim(0, 0.69)
        axis.set_yticks(y, labels)
        axis.grid(axis="x")
        axis.set_axisbelow(True)
    axes[0].invert_yaxis()
    fig.subplots_adjust(left=0.21, right=0.95, bottom=0.23, top=0.75, wspace=0.30)
    axes[0].legend(title="Training no-blink : blink", frameon=False, loc="upper left",
                   bbox_to_anchor=(-0.52, -0.18), ncol=2, fontsize=9, title_fontsize=9)
    fig.savefig(OUTPUT / "window_classification.png", dpi=180)
    plt.close(fig)


def continuous_chart(summary, audit):
    fig = plt.figure(figsize=(13, 7.6))
    grid = fig.add_gridspec(2, 2, width_ratios=(1.10, 1), height_ratios=(1, 1), hspace=0.60, wspace=0.32)
    tradeoff = fig.add_subplot(grid[:, 0])
    recordings = fig.add_subplot(grid[0, 1])
    failures = fig.add_subplot(grid[1, 1])
    fig.suptitle("BlinkIt on continuous EEG", x=0.075, y=0.97, ha="left", fontsize=23, weight="bold")
    fig.text(0.075, 0.913, "32 reviewed minutes · 229 annotated blinks · one Noor recording held out at a time",
             color=MUTED, fontsize=11)
    tradeoff.set_title("Precision and recall by configuration", loc="left", fontsize=13, weight="bold", pad=12)
    tradeoff.scatter([r["recall"] for r in summary], [r["precision"] for r in summary],
                     color=GRAY, edgecolor="#71879B", linewidth=0.6, s=34, label="Tested configurations")
    highlighted = [
        ("localized25", "cnn", "precision", BLUE, "Plain CNN", (-80, 15)),
        ("localized_hybrid25", ENSEMBLE, "precision", TEAL, "Ensemble", (-30, 17)),
        ("localized_hybrid25", ENSEMBLE, "f1", AMBER, "Ensemble, higher recall", (-115, 13)),
    ]
    for case, model, policy, color, label, offset in highlighted:
        r = next(r for r in summary if (r["experiment"], r["model"], r["policy"]) == (case, model, policy))
        tradeoff.scatter(r["recall"], r["precision"], color=color, s=100, edgecolor="white", linewidth=0.9,
                         zorder=6)
        tradeoff.annotate(label, (r["recall"], r["precision"]), xytext=offset,
                          textcoords="offset points", fontsize=9, color=color, weight="bold")
    tradeoff.set(xlim=(0, 0.80), ylim=(0, 0.54), xlabel="Blink recall", ylabel="Alert precision")
    tradeoff.xaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    tradeoff.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    tradeoff.grid(True)
    tradeoff.set_axisbelow(True)
    tradeoff.legend(loc="lower right", frameon=False, fontsize=9)

    recordings.set_title("Blinks detected by recording", loc="left", fontsize=13, weight="bold", pad=12)
    bars = np.arange(len(audit))
    detected = np.array([r["true_positive"] for r in audit])
    total = np.array([r["reviewed_blinks"] for r in audit])
    recordings.barh(bars, detected, color=TEAL, height=0.52, label="Detected")
    recordings.barh(bars, total - detected, left=detected, color=GRAY, height=0.52, label="Missed")
    recordings.set_yticks(bars, [r["session"].replace("noor", "Noor ") for r in audit])
    recordings.set_xlim(0, 96)
    recordings.invert_yaxis()
    for i, (n, d) in enumerate(zip(detected, total)):
        recordings.text(d + 2, i, f"{n}/{d}", va="center", fontsize=10, color=INK)
    recordings.set_xlabel("Reviewed blinks")
    recordings.grid(axis="x")
    recordings.set_axisbelow(True)
    recordings.legend(loc="lower right", bbox_to_anchor=(1, -0.39), frameon=False, ncol=2, fontsize=8)

    counts = [
        sum(r["away_from_labeled_events"] for r in audit),
        sum(r["extra_alert_for_blink"] + r["near_blink_outside_tolerance"] for r in audit),
        sum(r["inside_sustained_closure"] for r in audit),
    ]
    failures.set_title(f"{sum(counts)} false alarms from the selected ensemble", loc="left",
                       fontsize=13, weight="bold", pad=12)
    for i, (count, color) in enumerate(zip(counts, (BLUE, "#6380A2", AMBER))):
        failures.barh(i, count, color=color, height=0.52)
        failures.text(count + 3, i, str(count), va="center", color=INK, fontsize=10)
    failures.set_yticks(np.arange(3), ["Away from events", "Near blink / extra", "During closure"])
    failures.set_xlim(0, 132)
    failures.invert_yaxis()
    failures.set_xlabel("False alarms")
    failures.grid(axis="x")
    failures.set_axisbelow(True)
    fig.subplots_adjust(left=0.075, right=0.965, bottom=0.12, top=0.82)
    fig.savefig(OUTPUT / "continuous_detection.png", dpi=180)
    plt.close(fig)


def main():
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "text.color": INK, "axes.labelcolor": MUTED,
        "xtick.color": MUTED, "ytick.color": INK, "axes.spines.top": False,
        "axes.spines.right": False, "axes.edgecolor": "#B8C4D0", "grid.color": "#E7ECF0",
    })
    OUTPUT.mkdir(exist_ok=True)
    summary = summarize_continuous()
    audit = audit_alerts()
    write_csv(OUTPUT / "continuous_summary.csv", summary)
    write_csv(OUTPUT / "false_alarm_breakdown.csv", audit)
    window_chart()
    continuous_chart(summary, audit)


if __name__ == "__main__":
    main()
