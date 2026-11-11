# EEG recordings: conversion from OpenSignals .txt to Excel


This README describes how the raw BITalino / OpenSignals text recordings were turned into Excel workbooks, what each column means, and what to keep in mind when working with them.

## 1. Files

| # | Source (.txt) | Workbook (.xlsx) | Recorded | Samples | Duration | EEG channel |
|---|---|---|---|---|---|---|
| 1 | `noor_recording1.txt` | `noor_recording1.xlsx`   | 669,600 | 669.6 s (11.16 min) | 
| 2 | `rahma_recording1.txt` | `rahma_recording1.xlsx` | 760,950 | 760.95 s (12.68 min)|
| 3 | `noor_recording2.txt` | `noor_recording2.xlsx`   | 627,300 | 627.3 s (10.46 min) | 
| 4 | `noor_recording3.txt` | `noor_recording3.xlsx`   | 652,200 | 652.2 s (10.87 min) | 

All four were recorded with the same BITalino device (`98:D3:21:FC:8C:2A`) at **1000 Hz**, 10-bit resolution, one analog channel carrying the EEG.


## 2. How the conversion was done, and why

The conversion was done **through AI**:  wrote and ran a Python script (pandas) that read each .txt file, did the maths in code, and wrote the finished numbers into the .xlsx file.

The microvolt values are therefore **plain numbers in the cells, not Excel formulas.** This was a deliberate choice. Each recording has 600,000 to 760,000 rows, and putting the conversion formula in a cell for every row would make Excel slow to open, slow to scroll and very slow to recalculate. Computing it once outside Excel keeps the workbooks light and responsive. The original raw ADC values are kept next to the converted ones, so nothing is lost and you can recompute with a different formula if you need to (see section 5).

The same script and the same steps were used for every file (the script is included as `convert_opensignals_to_xlsx.py`).

### Steps

1. **Read the file.** Lines starting with `#` are the OpenSignals header (device, sampling rate, date, start time, channel layout). The remaining lines are tab-separated data with 11 columns: `nSeq, I1, I2, O1, O2, A1, A2, A3, A4, A5, A6`. Every row ends with a tab, which creates an empty 12th column that is ignored.
2. **Convert ADC counts to microvolts** with the BITalino EEG transfer function:

   `EEG (uV) = (ADC / 2^10 − 1/2) × 3.3 V / G × 1e6`, with `G = 40000`

   Results are rounded to 3 decimals.
4. **Make a continuous time base.** `nSeq` is a 4-bit counter that wraps from 15 back to 0, so it is useless as a time axis. It was replaced by `Sample` (0, 1, 2, ...) and `Time (s) = Sample / 1000`. In all four files the `nSeq` counter is continuous, which means **no samples were dropped** during recording.
5. **Keep or drop the other columns.** Channels with a little signal (tiny values from unconnected inputs) are kept as raw counts. Columns that are zero for the whole recording are dropped.
6. **Write the workbook** with two sheets, `Data` and `Info`. The header row on `Data` is frozen.

### What was not done

This is a **format conversion only**. No filtering, no baseline or DC removal, no notch filter, no resampling and no event detection was applied. The signal is exactly what the sensor recorded, scaled to µV.

## 3. What is in each workbook

**`Data` sheet**

| Column | Meaning |
|---|---|
| `Sample` | Continuous sample index starting at 0 |
| `Time (s)` | Seconds since the start of the recording (`Sample / 1000`) |
| `<channel> raw (ADC)` | Raw EEG value, 0 to 1023 (this data peaks at 1018) |
| `<channel> EEG (uV)` | The same value converted to microvolts |
| `A? raw` (extra columns) | Raw counts from the other analog inputs that show small values |

**`Info` sheet:** source file name, device, sampling rate, date, start time, comments, number of samples, duration, which channel is the EEG, the conversion formula, and which columns were dropped.

The extra columns only hold tiny values (never above 9 counts, against about 510 for the EEG channel). They are not EEG and are kept only so no data is thrown away.

## 4. Things to be careful about

- **Saturation (clipping).** The EEG sensor only measures about ±40 µV. Anything larger is cut off at the ends of the ADC range, which in these files are the raw values **0** and **1018** (about −41.25 and +40.77 µV with the formula used). Share of samples sitting at either limit:

  | Recording | Samples at a limit |
  |---|---|
  | 1 | about 4.0 % |
  | 2 | about 2.0 % |
  | 3 | about 1.6 % |
  | 4 | about 7.1 % |

  The true size of those excursions is lost. Treat flat stretches at the limits as "too large to measure", not as real values.
- **Gain value (40000 vs 41782).** The workbooks use a gain of **40000** (range about ±41.25 µV). The project's `blink_preprocess.py` uses the datasheet value **41782** (range about ±39.49 µV). The sensor's datasheet revision decides which one is right, and that is not stored in the recording files. The difference is a constant scale factor of about 4.3 % (µV values at 41782 are 0.957 times those in the workbooks), so shapes, timing and clipping do not change. If you combine these workbooks with the project's preprocessing output, use the same gain on both sides. The raw ADC column makes it easy to recompute.
- **Row limits and size.** Excel holds 1,048,576 rows per sheet, so each recording fits on one sheet. Files are 18 to 22 MB and can take 10 to 30 seconds to open.

## 5. Reproducing or changing the conversion

```
python convert_opensignals_to_xlsx.py recording.txt recording.xlsx
```

Requires Python with `pandas` and `xlsxwriter`. To use a different gain, change `GAIN` at the top of the script (for example `41782.0`) and run it again. To recompute inside Excel for a small selection of rows, use the raw column: `=(C2/1024-0.5)*3.3/40000*1000000`, where `C2` is the raw EEG cell and the gain is your chosen value.

For filtering, saturation masks, blink candidate detection and model-ready windows, use `blink_preprocess.py` from the project. It reads the original .txt files directly, so the workbooks are not needed for that step.
