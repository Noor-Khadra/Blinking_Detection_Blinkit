#!/usr/bin/env python3
"""
label_eeg_samples.py - give every EEG sample a label from the synced video labels.

label codes (priority when events overlap: blink > sustained closure > rejected candidate)
  0 = no event        clean background, use as negatives
  1 = blink           accepted blink, onset..offset
  2 = sustained       accepted sustained closure - eyes closed, not a blink: exclude from training
  3 = uncertain       rejected candidate (gaze / occlusion / ambiguous) - exclude from negatives

Usage: python label_eeg_samples.py Noor_recording1.xlsx labels_eeg_synced.xlsx --out recording1_eeg_labelled
"""
import argparse
import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("eeg_xlsx"); ap.add_argument("labels_xlsx"); ap.add_argument("--out", default="eeg_labelled")
ap.add_argument("--xlsx", action="store_true", help="also write an Excel copy")
a = ap.parse_args()

eeg = pd.read_excel(a.eeg_xlsx, sheet_name="Data", engine="calamine")
lab = pd.read_excel(a.labels_xlsx, sheet_name="Labels", engine="calamine")
for c in ("event_type", "review_status"):
    lab[c] = lab[c].astype(str).str.strip()
n = len(eeg)
code = np.zeros(n, np.int8)
eid = np.zeros(n, np.int32)

def paint(rows, value):
    """overwrite onset..offset (inclusive) of each event"""
    for _, r in rows.iterrows():
        s, e = max(int(r.onset_eeg_sample), 0), min(int(r.offset_eeg_sample) + 1, n)
        code[s:e] = value
        eid[s:e] = r.event_id

# lowest priority first, each layer overwrites the previous one
acc = lab.review_status == "accepted"
paint(lab[~acc], 3)
paint(lab[acc & (lab.event_type == "sustained_closure")], 2)
paint(lab[acc & (lab.event_type == "blink")], 1)

names = np.array(["none", "blink", "sustained_closure", "uncertain"])
eeg["label"] = code
eeg["label_name"] = names[code]
eeg["event_id"] = eid
eeg.to_csv(a.out + ".csv", index=False)
print(eeg.label_name.value_counts().rename("samples").to_frame().assign(seconds=lambda d: d.samples / 1000))
if a.xlsx:
    with pd.ExcelWriter(a.out + ".xlsx", engine="openpyxl") as w:
        eeg.to_excel(w, sheet_name="Data", index=False)
        pd.DataFrame({"label": [0, 1, 2, 3], "label_name": names,
                      "meaning": ["no event - use as negatives",
                                  "accepted blink (video onset..offset)",
                                  "accepted sustained closure - exclude from training",
                                  "rejected candidate (gaze/occlusion/ambiguous) - exclude from negatives"]}
                     ).to_excel(w, sheet_name="Legend", index=False)
print("wrote", a.out)
