# Rahma 1: making an unseen-person test set (Rahma Bezzaouia)

Every current model was trained and tuned on Noor and Rana recordings. Rahma 1
is the only BITalino recording of a different person, so it is the only way
to test generalization honestly (project guide: "Use unseen people or
conditions when claiming that the system generalizes").

The recording is not usable as ground truth yet. `reports/webcam_eeg_synchronization.md`
lists two blockers:

1. **Label problems.** Some labels overlap or duplicate each other, and some
   "blinks" are far too long (one is 71 s).
2. **No trustworthy clock map.** There is no capture log. The grid-search map
   matched 15/44 held-out blinks, and the shifted controls matched 6–12/44.

## Steps

| Step | Script | Output |
|---|---|---|
| 1. Audit the labels | `audit_labels.py` | `reports/rahma1_labels_audit.md`, `reports/rahma1_labels_flagged.csv` |
| 2. Fix the flagged labels against the eye-camera video | manual | `data/derived/noor/session4/rahma1_labels_cleaned.csv` |
| 3. List strong EEG dips to use as anchors | `fit_sync_anchors.py candidates` | printed table |
| 4. Match 6–10 of those dips to blinks in the video (start from `data/synchronization/inputs/rahma1_anchor_worksheet.csv`, which gives a 6 s search window in the video for each strong dip based on the old rough map) | manual | `data/synchronization/inputs/rahma1_anchors.csv` (`video_s,eeg_s`) |
| 5. Fit and validate the map | `fit_sync_anchors.py fit` | `data/synchronization/rahma1_anchor_metrics.json`, aligned events |
| 6. Score the frozen detector's blind alerts (`reports/blink_transfer/noor12_rana1_to_rahma/`) on the validated labels | — | first unseen-person precision, recall and timing error |

The map counts as validated only when the leave-one-out anchor error stays
under 100 ms and the match rate is clearly above every time-shifted control.
If it fails, report that as a limitation rather than using unaligned labels.

## Run

```sh
pip install numpy pandas scipy
python3 src/rahma_session/audit_labels.py data/derived/noor/session4/rahma1_labels_reviewed.csv --out reports/rahma1_labels
python3 src/rahma_session/fit_sync_anchors.py candidates --eeg data/eeg/raw/rahma_recording1.txt --channel A4
python3 src/rahma_session/fit_sync_anchors.py fit --eeg data/eeg/raw/rahma_recording1.txt --channel A4 \
    --anchors data/synchronization/inputs/rahma1_anchors.csv \
    --labels data/derived/noor/session4/rahma1_labels_cleaned.csv --out data/synchronization/rahma1_anchor
```

The raw EEG file `data/eeg/raw/rahma_recording1.txt` is on the `feat/eeg-preprocessing` branch.
