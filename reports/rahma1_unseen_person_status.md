# Rahma 1 as an unseen-person test: status (Rahma Bezzaouia)

## 1. Labels: done

| | Before | After video review |
|---|---:|---:|
| Accepted blinks | 80 | **157** |
| Sustained closures | 0 | 16 |
| Near-duplicate labels | 11 | 0 |
| Overlapping labels | 2 | 0 |
| "Blinks" longer than 1.5 s (max 71.6 s) | 3 | 0 |
| Labels needing a video check | 28 | 1 (a deliberately kept slow blink, 0.83 s) |

How the labels were fixed:

- The 28 audited problems were checked one by one in the review page.
- The review page's own eye-openness trace (EAR) showed 111 eye closures with
  no label. All 111 were reviewed in the video: 101 were real blinks and were
  accepted, 10 were rejected (looking down or partial closures). 8 more blinks
  were added by hand.
- Four leftover duplicate groups were merged, keeping the label whose duration
  was closest to a typical blink. Five small overlaps were split at the
  midpoint between their two peaks.

The original labels missed about half of the blinks. Any metric computed
against them would have counted correct detections as false alarms.

Files: `data/derived/noor/session4/rahma1_labels_cleaned.csv`,
`reports/rahma1_labels_cleaned_audit.md`.

## 2. Model on an unseen person: first result (count level)

The team's live detector (`cnn_01` ensemble, threshold 0.8, the same
`StreamDetector`/`PeakGate` as the live app) was run over the full Rahma 1 EEG.
The model never saw Rahma's data. Alerts: `reports/rahma1_cnn01_alerts.csv`.
Reproduce with `src/rahma_session/run_live_model.py`.

| | Count | Duration | Rate |
|---|---:|---:|---:|
| Model alerts (EEG) | 207 | 12.7 min | 16.3 / min |
| Labeled blinks (video) | 157 | 11.3 min | 13.9 / min |

The rates are in the same range: the model fires about 17% more often than
Rahma actually blinked. Without a trusted clock map, alerts cannot be matched
to blinks one by one, so precision and recall on this person are **not
measured yet**.

These numbers were computed with a NumPy port of the checkpoint, because PyTorch
was unavailable in the analysis environment. The port reproduces the team's saved
test-window scores (`reports/rana_cnn_xgb/cnn_01_test_windows.csv`) to within
3e-7. The team should rerun `run_live_model.py` with PyTorch to confirm.

## 3. Video→EEG clock: still not validated

With the complete labels, a grid search (rate 0.90–1.10, offset ±150 s) finds a
single linear map (rate ≈ 1.053, offset ≈ 40.5 s). Under it, 111 of 157 labeled
blinks land within 150 ms of an EEG dip, against 33–47 for time-shifted
controls. But a split-half check fails:

| Fit on | Test on | Matched | Shifted controls |
|---|---|---:|---:|
| First half of blinks | Second half | 36 / 79 | 22–36 |
| Second half | First half | 10 / 78 | 13–22 |

One straight-line map does not hold across the whole recording. The most likely
cause is that the video's frame timing drifts, since there is no
capture-timestamp log for this session. Until it is fixed, per-blink scoring
would be unreliable, so it is not reported.

**Next step:** manual anchors spread over the recording (the
`fit_sync_anchors.py` workflow), or a piecewise map fitted and checked per
segment.

**Lesson for future recordings:** log capture timestamps, and start and end each
session with 3 forceful blinks as sync markers.
