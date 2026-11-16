# Webcam and EEG synchronization

Run from the repository root with Python 3.10 or later.

```sh
python3 -m pip install -r src/eeg_preprocessing/requirements-sync.txt
git fetch origin feat/eeg-preprocessing
python3 src/eeg_preprocessing/synchronize_webcam.py
```

The script reads the four OpenSignals text files and workbooks from `data/eeg/`
or from `origin/feat/eeg-preprocessing` by default. The audited inputs on that
branch are from commit `43f284c`. Use `--eeg-ref 43f284c` to pin that revision
or `--session noor3` to regenerate one recording. Each run writes
`<session>_aligned_events.csv` and `<session>_diagnostics.json` under
[`data/synchronization/`](../data/synchronization/).

## Timing inputs and verification

| Session | OpenSignals text file | Channel | EEG rows | Video timing |
| --- | --- | --- | ---: | --- |
| Noor 1 | `noor_recording1.txt` | A3 | 669,600 | Playback. No matching capture log was available. Reviewed events 76 and 77 use the corrected boundaries from the EEG branch. |
| Noor 2 | `noor_recording2.txt` | A4 | 627,300 | 18,184 elapsed capture timestamps from the Noor 2 frame timeline. First capture wall clock 2026-10-25 19:10:58.575523, 13.264477 s before EEG start. The 606.100 s AVI playback timeline spans 636.389 s of capture. |
| Noor 3 | `noor recording_3.txt` | A2 | 652,200 | 18,966 elapsed timestamps from the Noor 3 webcam capture log. The original one-based frame counter was converted to zero-based label indices. First capture wall clock 2026-11-11 12:12:24.987878, 2.801122 s before EEG start. The 632.167 s AVI playback timeline spans 649.617 s of capture. |
| Rahma 1 | `rahma_recording1.txt` | A4 | 760,950 | Playback. The reviewed export from GitHub commit `83db882` matches the Rahma review package's 30 fps indexing and 20,397-frame extent. No capture log could be tied to this recording. |

Each EEG header specifies 1,000 Hz. Every workbook data row was compared with
its original ADC row, zero-based index and nominal sample time. All comparisons
passed. The first, second and Rahma workbooks display legacy A2 headers on data
columns even though their ADC values match the channels above. The raw packet
counter was continuous in all four recordings. A 4-bit counter cannot reveal
losses of an exact multiple of 16 samples on its own.

## Fit and held-out check

EEG time in seconds is `rate * source_s + offset_s`. Every third accepted,
completed, non-overlapping blink in time order was used to estimate the fit.
The other eligible blinks were held out. An order-three, zero-phase 0.5–12 Hz
band-pass was applied to a copy of the raw ADC to find prominent negative dips.
The script applies the same map to each event's onset, peak and offset. Sample
indices refer to the raw recording timeline. The residual is the detected EEG
dip time minus the projected webcam peak. Shifted held-out controls compare
against chance agreement in an EEG recording with many candidate dips.

| Session | Rate | Offset (s) | Held-out within 150 ms | Unique | Median absolute / p95 (ms) | Raw clipping fraction |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Noor 1 | 0.999912812 | −17.391060 | 35/41 | 35 | 20.6 / 34.5 | 4.05% |
| Noor 2 | 1.000077762 | −12.686507 | 52/54 | 51 | 17.4 / 77.3 | 1.64% |
| Noor 3 | 1.000056960 | −2.210866 | 55/57 | 55 | 16.6 / 100.6 | 7.12% |
| Rahma 1 | 0.952755970 | +63.522156 | 15/44 | 15 | 49.2 / 132.1 | 2.01% |

Residual summaries cover matched peaks only. Noor 1 and Noor 2 use search
windows around the prior estimates and these metrics come from the rerun.
Each event row carries a match status and signal diagnostics from a
500 ms raw window around its projected peak. The clipping check uses the
recorded rail values 0 and 1018. Visual review should start with missed or shared
dips, overlapping labels, unusually long blinks and rail-heavy windows,
especially in Noor 3. Keep the three Noor maps provisional until that review.

Rahma's candidate fit has a near-tied alternative on the estimation subset and
matches only 15 held-out events. Shifted controls matched 6–12 of 44. Thirteen
accepted labels overlap, and event 58 is marked as a blink with a duration
above 71 seconds. Review those labels and obtain a session-matched capture
log or several confirmed shared anchors across the recording. Its exported EEG
times and sample indices represent an **unvalidated candidate** for review.

Blink and sustained closure retain their original classes. Zero negative
examples were emitted because confirmed continuously reviewed video intervals
were unavailable. The alignment filter is specific to dip detection. Training
preprocessing and training labels remain a separate decision.
