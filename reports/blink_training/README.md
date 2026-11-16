# BlinkIt training pilot

## Architecture choice

A single-channel frontal EEG study demonstrated rule-based blink detection
using digital filters in 24 participants. BLINKER also uses ocular waveform
properties such as amplitude, velocity and duration. These motivate an
amplitude rule and a regularized logistic regression on waveform features.
A 2025 study by Rejer and Gago validated CNN-based blink event detection
across ten subjects in three experimental setups. That motivates a compact
temporal CNN for BlinkIt. The experiments differ in participants, equipment
and evaluation, so their published scores are outside the BlinkIt comparison.

- [Single prefrontal channel study](https://doi.org/10.1016/j.cmpb.2015.10.011)
- [BLINKER](https://doi.org/10.3389/fnins.2017.00012)
- [Rejer and Gago's CNN study](https://doi.org/10.1088/1741-2552/add49d)
- [EEGNet](https://doi.org/10.1088/1741-2552/aace8c)
- [EEG leakage and subject-based holdout](https://doi.org/10.3389/fnins.2024.1373515)

The CNN uses two temporal convolution layers plus max and mean pooling,
with 1,297 trainable parameters. It is small enough to compare directly with
the baselines. The single electrode signal gives EEGNet's spatial filters
little to learn across channels. The available 90 positive windows from one
person also favor small models.

## Data and split

The pilot uses the 1,415 prepared windows from dataset commit `dcd5431`.
The one-second windows are resampled from 1,000 to 100 Hz using polyphase
antialias filtering. The index file supplies the recording and class.
Only the waveform enters the model.

Each fold trains on two sessions and holds out the third. The two training
sessions also serve as alternating validation sessions to choose thresholds.
Background windows are sampled without replacement within each training
session. Validation and held-out sets retain all eligible windows. The
13 sustained closures are excluded from binary fitting and reported as a
separate challenge set.

| Test recording | Train blinks | Train no-blink (3:1) | Test blinks | Test no-blink | Test closures |
| --- | ---: | ---: | ---: | ---: | ---: |
| Noor 1 | 53 | 159 | 37 | 490 | 5 |
| Noor 2 | 45 | 135 | 45 | 405 | 5 |
| Noor 3 | 82 | 246 | 8 | 417 | 3 |

## Results

| Model | Average precision, 3:1 | F1, 3:1 | Average precision, 2:1 | F1, 2:1 |
| --- | ---: | ---: | ---: | ---: |
| Amplitude threshold | 0.518 | 0.406 | 0.518 | 0.406 |
| Waveform logistic regression | 0.531 | 0.230 | 0.497 | 0.319 |
| Small 1D CNN | 0.576 | 0.435 | 0.556 | 0.409 |

Scores are means over test recordings and, for the CNN, three initialization
seeds. The ratios refer to training data and leave the held-out data intact.
Average precision summarizes the ranking across thresholds. F1 uses the
threshold chosen from training-session validation. The amplitude rule has
no fitted classifier, so its results are independent of the training ratio.

With 3:1 sampling, the CNN detected 37/37 Noor 1 blinks with 56–84 false
positives across the seeds, 30–31/45 on Noor 2 with 3–7 false positives, and
0–1/8 on Noor 3 with 22–36 false positives. It flagged 4–5 of 13 sustained
closures as blinks across the three seeds. Noor 3 also had higher background
amplitude and a different dominant polarity among retained blink windows.
Those are observations on this small pilot and motivate checking the signal
and testing polarity robustness on development data.

## Next experiments

1. Keep the amplitude model as a baseline and the compact CNN as the
   leading candidate. Compare polarity augmentation and rolling background
   normalization on development recordings. Keep their parameters frozen for
   the next independent test.
2. Once Rahma's EEG labels and review coverage are reliable, develop on
   Noor and test once on Rahma with the same frozen preprocessing and decision
   threshold. One held-out person tests transfer to that person. Broader
   generalization requires more participants.
3. Evaluate a sliding detector on continuously reviewed EEG with event-level
   precision, recall, latency and false alarms per minute. This pilot uses
   quality-filtered, event-centered positive windows, with positive inclusion
   partly dependent on a recognizable EEG dip. The results describe that
   window benchmark. Continuous evaluation should include every reliably
   aligned reviewed blink and score matched events after merging repeated
   window detections.

## Reproduce

From the repository root:

```sh
python3 -m pip install -r src/model/requirements.txt
python3 src/model/run_blink_pilot.py
python3 src/model/run_blink_pilot.py --negative-ratio 2 --output reports/blink_training/ratio2
```

`splits.json` contains zero-based dataset row indices for the outer and
threshold-selection folds. `metrics.json` contains the scores and decision
thresholds. `predictions.csv` maps each held-out prediction to its input row.
The JSON files include checksums of the dataset and index used for the run.
