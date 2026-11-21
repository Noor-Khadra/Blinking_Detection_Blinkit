"""Build one table describing every EEG recording in the repo (author: Rahma Bezzaouia).

For each session it reports who/what it is, duration, signal quality, how the
blink labels were made, how many there are, and how well the webcam clock was
matched to the EEG. Every number is read from files already in the repo.

    python3 src/data_inventory/build_inventory.py
writes reports/data_inventory/inventory.csv and reports/data_inventory/sections.csv
"""
import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "data_inventory"
SYNC = ROOT / "data" / "synchronization"


def old_session(name, person, labels_path, label_note, extra=None):
    d = json.loads((SYNC / f"{name}_diagnostics.json").read_text())
    ho, ctrl = d["held_out"], d["held_out_shifted_controls"]
    labels = pd.read_csv(ROOT / labels_path)
    acc = labels[labels.get("review_status", "accepted").eq("accepted")] if "review_status" in labels else labels
    row = {
        "session": name, "person": person, "date": d["eeg_start_local_time"][:10],
        "eeg_minutes": round(d["observed_samples"] / d["sampling_rate_hz"] / 60, 1),
        "channel": d["channel"], "clipping_pct": round(100 * d["raw_clipping_fraction"], 1),
        "missing_samples": d["minimum_missing_samples"], "protocol": "unscripted",
        "label_method": label_note,
        "blinks": int(acc.event_type.eq("blink").sum()),
        "sustained_closures": int(acc.event_type.eq("sustained_closure").sum()),
        "clock_method": "linear map fitted on every 3rd reviewed blink"
                        + (" + capture log" if d.get("capture_elapsed_s") else ""),
        "sync_check": f"held-out blinks within 150 ms: {ho['within_150ms']}/{ho['total']}; "
                      f"shifted controls {min(c['within_150ms'] for c in ctrl)}-{max(c['within_150ms'] for c in ctrl)}",
        "sync_status": "validated" if ho["within_150ms"] > 1.5 * max(c["within_150ms"] for c in ctrl)
                       else "NOT validated",
    }
    row.update(extra or {})
    return row


def protocol_session(name, person, model_dir):
    meta = json.loads((ROOT / f"data/raw/datasets/{name}/session.json").read_text())
    ev = pd.read_csv(ROOT / f"data/model/{model_dir}/events.csv")
    ev = ev[ev.session.eq(name)]
    chk = pd.read_csv(ROOT / f"data/model/{model_dir}/{name}/alignment_checks.csv")
    eeg = pd.read_csv(ROOT / f"data/raw/datasets/{name}/eeg.csv")
    adc = eeg[[c for c in eeg.columns if c.lower() in ("adc", "a4", "raw", "value")][0]]
    clipping = round(100 * ((adc <= 0) | (adc >= 1018)).mean(), 1)
    return {
        "session": name, "person": person, "date": meta["session_start_utc"][:10],
        "eeg_minutes": round(meta["eeg_rows"] / meta["eeg_sampling_rate_hz"] / 60, 1),
        "channel": meta["eeg_channel"], "clipping_pct": clipping,
        "missing_samples": meta["estimated_missing_eeg_samples"],
        "protocol": "8-section scripted (rest, reading, video, talking, voluntary blinks, "
                    "eye movements, artifacts, eyes closed)",
        "label_method": "automatic MediaPipe EAR, unreviewed (" + ev.label_source.iloc[0] + ")",
        "blinks": int(ev.event_type.eq("blink").sum()),
        "sustained_closures": int(ev.event_type.eq("sustained_closure").sum()),
        "clock_method": "host receipt clocks logged during capture (no hardware sync)",
        "sync_check": f"labeled blinks with an EEG extremum within 150 ms: "
                      f"{(chk.extrema_within_150ms > 0).sum()}/{len(chk)}",
        "sync_status": "logged (provisional)",
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    rows = [
        old_session("noor1", "Noor", "data/derived/noor/session1/labels.csv", "webcam EAR + human review"),
        old_session("noor2", "Noor", "data/derived/noor/session2/labels.csv", "webcam EAR + human review"),
        old_session("noor3", "Noor", "data/derived/noor/session3/labels.csv", "webcam EAR + human review"),
    ]
    rahma = old_session("rahma1", "Rahma", "data/derived/noor/session4/rahma1_labels_reviewed.csv",
                        "webcam EAR + human review")
    cleaned = ROOT / "data/derived/noor/session4/rahma1_labels_cleaned.csv"
    if cleaned.exists():
        c = pd.read_csv(cleaned)
        c = c[c.review_status.eq("accepted")]
        rahma.update({"label_method": "webcam EAR + full human re-review (missed blinks added, duplicates merged)",
                      "blinks": int(c.event_type.eq("blink").sum()),
                      "sustained_closures": int(c.event_type.eq("sustained_closure").sum())})
    rows.append(rahma)
    rows += [protocol_session("rana_01_clean", "Rana", "rana_new"),
             protocol_session("rana_02", "Rana", "rana_new"),
             protocol_session("noor4-clean", "Noor", "noor4_new")]
    pub = json.loads((ROOT / "data/model/rana1/dataset_summary.json").read_text())
    rows.append({"session": "rana1 (alias)", "person": "public EEGEyeNet participant EP10 (not Rana)",
                 "date": None, "eeg_minutes": None, "channel": "E22 exported as A3", "clipping_pct": None,
                 "missing_samples": None, "protocol": "public dataset, eye-tracking task",
                 "label_method": "eye-tracker annotations from the public dataset",
                 "blinks": pub.get("blink_windows", 56), "sustained_closures": 0,
                 "clock_method": "shared timeline in source data", "sync_check": "n/a",
                 "sync_status": "source-synchronized (different device)"})
    for r in rows:
        if r["eeg_minutes"]:
            r["blinks_per_min"] = round(r["blinks"] / r["eeg_minutes"], 1)
    inv = pd.DataFrame(rows)
    inv.to_csv(OUT / "inventory.csv", index=False)

    proto = pd.read_csv(ROOT / "data/raw/datasets/rana_02/protocol.csv")
    sec_rows = []
    for name, model_dir in [("rana_01_clean", "rana_new"), ("rana_02", "rana_new"), ("noor4-clean", "noor4_new")]:
        ev = pd.read_csv(ROOT / f"data/model/{model_dir}/events.csv")
        ev = ev[ev.session.eq(name) & ev.event_type.eq("blink")]
        sec = pd.cut(ev.peak_session_s, bins=[0] + proto.end_s.tolist(), labels=proto.section.tolist())
        counts = sec.value_counts()
        for _, p in proto.iterrows():
            minutes = (p.end_s - p.activity_start_s) / 60
            n = int(counts.get(p.section, 0))
            sec_rows.append({"session": name, "section": p.section, "activity": p.activity,
                             "auto_blinks": n, "per_min": round(n / minutes, 1)})
    pd.DataFrame(sec_rows).to_csv(OUT / "sections.csv", index=False)
    print(inv.drop(columns=["protocol", "clock_method"]).to_string(index=False))


if __name__ == "__main__":
    main()
