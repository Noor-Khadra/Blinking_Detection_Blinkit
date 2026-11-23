"""Can BlinkIT's detector control music? Evidence before the demo (author: Rahma Bezzaouia).

Three questions, each answered from data already in the repo:

A. Spontaneous blinks: how often does natural blinking accidentally look like a
   2/3/4-blink command?  Input: human-reviewed webcam labels (Noor 1-3), i.e. a
   perfect detector.
B. Real detector, no gestures: how many false commands per minute does the
   recognizer produce on the held-out detector alerts in
   reports/blink_accuracy/clipping45/alerts.csv (boosted model, F1 policy)?
C. Deliberate gestures: if a user performs a command, how often is it recognized
   correctly, missed, or turned into the WRONG command?  Monte Carlo simulation
   using the detector's measured recall and false-alert rate (no gesture
   recordings exist yet, so this is a prediction to be tested on demo day).

Outputs go to reports/blink_control/ (results.json, gesture_sweep.csv,
confusion.csv, gesture_tradeoff.png).

Run from the repo root:
    python3 src/blink_control/evaluate_gestures.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from gestures import COMMANDS, recognize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "blink_control"
GAPS = [0.5, 0.6, 0.7, 0.8, 1.0, 1.2]

# Detector operating points reported by the team
DETECTORS = {
    # reports/rana_noor2_replay/existing_gate_test.csv, "faster_matched_budget" row
    "CNN Rana+Noor2 replay": {"recall": 0.742, "false_per_min": 4.02},
    # reports/blink_accuracy/README.md, boosted model, 45% clipping gate
    "Boosted model, Noor held-out": {"recall": 0.668, "false_per_min": 2.68},
}


def spontaneous(gap):
    ev = pd.read_csv(ROOT / "data/model/webcam_events.csv")
    cov = pd.read_csv(ROOT / "data/model/reviewed_coverage.csv").set_index("session")
    rows = []
    for s in ["noor1", "noor2", "noor3"]:
        t = ev[(ev.session == s) & (ev.event_type == "blink")].peak_s.to_numpy()
        minutes = (cov.loc[s, "end_sample"] - cov.loc[s, "start_sample"]) / 1000 / 60
        cmds = recognize(t, max_gap=gap)
        rows.append({"session": s, "minutes": minutes, "blinks": len(t),
                     **{c: sum(x["command"] == c for x in cmds) for c in COMMANDS.values()}})
    return pd.DataFrame(rows)


def detector_alerts(gap):
    a = pd.read_csv(ROOT / "reports/blink_accuracy/clipping45/alerts.csv")
    a = a[(a.model == "boost_mined") & (a.policy == "f1")]
    cov = pd.read_csv(ROOT / "data/model/reviewed_coverage.csv").set_index("session")
    rows = []
    for s, g in a.groupby("test"):
        minutes = (cov.loc[s, "end_sample"] - cov.loc[s, "start_sample"]) / 1000 / 60
        cmds = recognize(g.detection_sample.to_numpy() / 1000, max_gap=gap)
        rows.append({"session": s, "minutes": minutes, "alerts": len(g),
                     **{c: sum(x["command"] == c for x in cmds) for c in COMMANDS.values()}})
    return pd.DataFrame(rows)


def simulate(n_blinks, recall, false_per_min, gap, trials=4000, spacing=0.5, jitter=0.08, seed=0):
    """Intended gesture of n blinks; each detected with prob `recall`; random
    false alerts in a 6 s window around it. Returns outcome counts."""
    rng = np.random.default_rng(seed + n_blinks)
    outcomes = {"correct": 0, "missed": 0, "wrong": 0}
    window = 6.0
    for _ in range(trials):
        start = 2.0
        true_t = start + np.cumsum(np.r_[0, rng.normal(spacing, jitter, n_blinks - 1)])
        detected = true_t[rng.random(n_blinks) < recall]
        k = rng.poisson(false_per_min * window / 60)
        fake = rng.uniform(0, window, k)
        times = np.sort(np.r_[detected, fake])
        # the detector never reports two alerts closer than 0.35 s
        kept = []
        for t in times:
            if not kept or t - kept[-1] >= 0.35:
                kept.append(t)
        cmds = recognize(kept, max_gap=gap)
        intended = COMMANDS[n_blinks]
        got = [c["command"] for c in cmds if c["end_s"] >= start - 0.5]
        if not got:
            outcomes["missed"] += 1
        elif got[0] == intended:
            outcomes["correct"] += 1
        else:
            outcomes["wrong"] += 1
    return {k: v / trials for k, v in outcomes.items()}


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    sweep = []
    for gap in GAPS:
        sp, de = spontaneous(gap), detector_alerts(gap)
        for name, df in [("spontaneous_labels", sp), ("detector_alerts", de)]:
            mins = df.minutes.sum()
            row = {"source": name, "max_gap_s": gap, "minutes": round(mins, 1)}
            for c in COMMANDS.values():
                row[f"{c}_per_10min"] = round(10 * df[c].sum() / mins, 2)
            row["any_false_command_per_10min"] = round(10 * df[list(COMMANDS.values())].sum().sum() / mins, 2)
            sweep.append(row)
    sweep = pd.DataFrame(sweep)
    sweep.to_csv(OUT / "gesture_sweep.csv", index=False)

    conf = []
    for det, p in DETECTORS.items():
        for gap in GAPS:
            for n in COMMANDS:
                r = simulate(n, p["recall"], p["false_per_min"], gap)
                conf.append({"detector": det, "max_gap_s": gap, "gesture": COMMANDS[n], "blinks": n,
                             **{k: round(v, 3) for k, v in r.items()}})
    conf = pd.DataFrame(conf)
    conf.to_csv(OUT / "confusion.csv", index=False)

    chosen = 0.8
    per_session = {}
    for name, df in [("spontaneous_labels", spontaneous(chosen)), ("detector_alerts", detector_alerts(chosen))]:
        df["false_commands_per_10min"] = (10 * df[list(COMMANDS.values())].sum(axis=1) / df.minutes).round(2)
        per_session[name] = df.round(2).to_dict(orient="records")
        print(name); print(df.round(2).to_string(index=False)); print()
    results = {
        "per_session_at_chosen_gap": per_session,
        "gesture_map": {str(k): v for k, v in COMMANDS.items()},
        "chosen_max_gap_s": chosen,
        "false_commands_per_10min_at_chosen_gap": sweep[sweep.max_gap_s == chosen]
            .set_index("source")["any_false_command_per_10min"].to_dict(),
        "deliberate_gesture_outcomes_at_chosen_gap": conf[conf.max_gap_s == chosen]
            .drop(columns="max_gap_s").to_dict(orient="records"),
        "command_latency_note": "time from last blink to command = max_gap (burst must end) + "
                                "detector delay (~0.95 s replay, excluding Bluetooth/audio)",
        "caveats": [
            "Spontaneous-blink rates use Noor 1-3 only (one person, ~32 min). The labels do not mark which "
            "blinks were voluntary; Noor 2 and 3 contain many close blink pairs (20 and 16 gaps < 0.8 s vs 2 in "
            "Noor 1), so if those sessions included a voluntary-blink section, the pooled rate overestimates "
            "accidental commands. Noor 1 is the cleanest estimate of natural blinking.",
            "Detector alerts come from the boosted model; the newer CNN has no per-alert file in the repo.",
            "Gesture outcomes are simulated from overall recall; deliberate blinks are usually stronger "
            "than spontaneous ones, so real recall on gestures may be higher. Must be measured.",
        ],
    }
    (OUT / "results.json").write_text(json.dumps(results, indent=2))
    plot(sweep, conf)
    print(sweep.to_string(index=False))
    print()
    print(conf[conf.max_gap_s == chosen].to_string(index=False))


def plot(sweep, conf):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    for src, style in [("spontaneous_labels", "o--"), ("detector_alerts", "s-")]:
        d = sweep[sweep.source == src]
        label = "natural blinks (webcam labels)" if src == "spontaneous_labels" else "real detector alerts"
        ax[0].plot(d.max_gap_s, d.any_false_command_per_10min, style, label=label)
    ax[0].set_xlabel("max gap between blinks in a command (s)")
    ax[0].set_ylabel("unwanted commands per 10 min")
    ax[0].set_title("Cost: accidental commands (no gestures made)")
    ax[0].legend(frameon=False)
    d = conf[(conf.detector == "CNN Rana+Noor2 replay") & (conf.max_gap_s == 0.8)]
    x = np.arange(len(d))
    ax[1].bar(x, d.correct, label="correct", color="#2a9d8f")
    ax[1].bar(x, d.wrong, bottom=d.correct, label="wrong command", color="#e76f51")
    ax[1].bar(x, d.missed, bottom=d.correct + d.wrong, label="missed", color="#bbbbbb")
    ax[1].set_xticks(x, [f"{g}\n({b} blinks)" for g, b in zip(d.gesture, d.blinks)])
    ax[1].set_ylim(0, 1)
    ax[1].set_ylabel("fraction of attempts")
    ax[1].set_title("Benefit: deliberate gestures (simulated, CNN recall 74%)")
    ax[1].legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "gesture_tradeoff.png", dpi=160)


if __name__ == "__main__":
    main()
