# BlinkIt model inputs

| File | Contents |
| --- | --- |
| `webcam_events.csv` | Accepted, completed blink and sustained-closure events. Includes the class and zero-based onset, peak and offset frames with video times in seconds. |
| `eeg_events.csv` | Provided EEG event annotations. Includes original zero-based EEG sample indices and an eligibility flag. Joins to webcam events by recording and event ID. |
| `reviewed_coverage.csv` | EEG sample ranges covered by continuous webcam review. |
| `eeg_windows.npz` | NumPy arrays `X` (float32, one row per 1,000-sample EEG window) and `y` (integer class labels). |
| `window_index.csv` | Row-for-row window metadata, including subject, recording, event and the original sample range. The range includes the start and excludes the end. |
| `dataset_summary.json` | Class codes, per-session counts and preparation exclusions. |

The preparation code drops rejected, incomplete and unsupported event labels
and exact duplicate event spans. It assigns consecutive one-based IDs within
each session after sorting by onset time. The dataset uses EEG as the model
input. Webcam fields, IDs and recording metadata are available for review and
grouped evaluation.

EEG examples use the original 1,000 Hz sampling rate. Each window is
median-centered and scaled by 512 ADC counts. The script filters out
ineligible annotations, windows intersecting another annotated event, sample
gaps, flat signals and windows with more than 10% of samples at the recorded
ADC limits. It keeps blinks and sustained closures as separate classes.

Class codes are 0 for no blink, 1 for blink and 2 for sustained closure. This
release includes 1,312 no-blink, 90 blink and 13 sustained-closure windows from
the three Noor sessions. The reviewer confirmed that the videos were watched
throughout and missed closures were added to the labels. Each negative window
lies within reviewed coverage and stays at least 150 ms away from annotated
events. The same EEG quality checks apply to all classes.

Rahma 1 contributes reviewed webcam labels, with EEG examples pending usable
event annotations. The EEG examples come from one participant. Within Noor,
group by session for an exploratory split and keep session metadata out of the
model features. When training a binary detector, handle the class imbalance
within the training split and evaluate blinks and sustained closures distinctly.

## Rebuild the EEG windows

```sh
python3 -m pip install 'numpy>=1.26'
git fetch origin feat/eeg-preprocessing
python3 src/eeg_preprocessing/prepare_model_data.py windows
```

The builder uses raw recordings available under `data/eeg/raw/` or on
`origin/feat/eeg-preprocessing`. Use `--raw-dir` to read raw OpenSignals text
from another directory and `--eeg-ref 43f284c` to pin the source revision.
The default build uses `reviewed_coverage.csv` alongside the EEG event table.
Pass `--reviewed-coverage path.csv` to supply other continuously reviewed
intervals as `session,start_sample,end_sample` rows on the EEG sample
timeline. The end index is exclusive. A custom EEG event table requires an
explicit coverage file to generate no-blink examples.

To clean a new export, place the reviewed CSV for each session in an input
directory. Use the file names `noor1.csv`, `noor2.csv`, `noor3.csv` and
`rahma1.csv`. The cleaner accepts frame/time columns from the webcam exports
and optional `onset_sample,peak_sample,offset_sample,use_for_training` from
provided EEG annotations. Sessions can be processed with webcam labels alone.

```sh
python3 src/eeg_preprocessing/prepare_model_data.py --output prepared-data clean --source-dir reviewed-exports
```

```python
import numpy as np

with np.load("data/model/eeg_windows.npz") as data:
    X, y = data["X"], data["y"]
```
