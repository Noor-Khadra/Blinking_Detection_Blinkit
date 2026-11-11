#!/usr/bin/env python3
"""
convert_opensignals_to_xlsx.py
OpenSignals (BITalino) .txt  ->  .xlsx   (Data sheet + Info sheet)

Usage:  python convert_opensignals_to_xlsx.py recording.txt recording.xlsx

Assumption: the EEG is always on channel A2 (EEG_CHANNEL below). Every other column is dropped.

Method:
  1. Parse the '#' header (JSON) and the tab-separated data block (11 columns; the 12th,
     empty column created by the trailing tab on each row is ignored).
  2. Take channel A2 and convert it from ADC counts to microvolts:
         uV = (ADC / 2^10 - 1/2) * 3.3 V / G * 1e6        with G = 40000
  3. Replace nSeq (4-bit counter that wraps 0-15) with a continuous Sample index and
     Time (s) = Sample / sampling rate.
  4. Write a Data sheet (Sample, Time, A2 raw, A2 EEG in uV) and an Info sheet.
"""
import json
import os
import sys

import pandas as pd

COLS = ["nSeq", "I1", "I2", "O1", "O2", "A1", "A2", "A3", "A4", "A5", "A6"]
EEG_CHANNEL = "A2"
VCC, GAIN, BITS = 3.3, 40000.0, 10


def convert(src, dst):
    with open(src) as fh:
        meta = list(json.loads(fh.readlines()[1][2:]).values())[0]
    df = pd.read_csv(src, comment="#", sep="\t", header=None, usecols=range(11), names=COLS)
    fs = meta["sampling rate"]
    eeg = EEG_CHANNEL

    # Safety check: a live EEG channel rests near mid-scale (~512 of 1023).
    mean = df[eeg].mean()
    if not 400 <= mean <= 620:
        print(f"WARNING: {eeg} mean is {mean:.1f}, not near mid-scale (~512). "
              f"This recording's EEG may be on another channel.", file=sys.stderr)

    out = pd.DataFrame({"Sample": range(len(df))})
    out["Time (s)"] = out["Sample"] / fs
    out[f"{eeg} raw (ADC)"] = df[eeg]
    out[f"{eeg} EEG (uV)"] = ((df[eeg] / 2**BITS - 0.5) * VCC / GAIN * 1e6).round(3)

    info = pd.DataFrame({
        "Field": ["Source file", "Device", "Sampling rate (Hz)", "Recording date", "Start time",
                  "Comments", "Samples", "Duration (s)", "Channel used for EEG",
                  f"Conversion {eeg} -> uV", "Dropped columns", "Sample / Time columns"],
        "Value": [os.path.basename(src) + " (OpenSignals text)",
                  f"BITalino ({meta['device name']})", fs, meta["date"], meta["time"],
                  meta["comments"] or "(none)", len(df), len(df) / fs,
                  f"{eeg} (mean ~{mean:.0f} = mid-scale)",
                  "uV = (ADC/2^10 - 1/2) * 3.3 V / 40000 * 1e6  (BITalino EEG transfer function, gain 40000)",
                  "Everything except A2: nSeq, I1, I2, O1, O2, A1, A3, A4, A5, A6",
                  f"Sample = continuous index (replaces nSeq); Time = Sample / {fs}"],
    })

    with pd.ExcelWriter(dst, engine="xlsxwriter") as w:
        out.to_excel(w, sheet_name="Data", index=False)
        info.to_excel(w, sheet_name="Info", index=False)
        w.sheets["Data"].freeze_panes(1, 0)
        w.sheets["Data"].set_column(0, len(out.columns) - 1, 14)
        w.sheets["Info"].set_column(0, 0, 24)
        w.sheets["Info"].set_column(1, 1, 90)
    return out


if __name__ == "__main__":
    out = convert(sys.argv[1], sys.argv[2])
    print(sys.argv[2], len(out), "rows; EEG on", EEG_CHANNEL, "; all other columns dropped")