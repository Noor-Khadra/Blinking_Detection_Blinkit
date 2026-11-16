# Comparing threshold-calibration splits

Each run still holds out one complete Noor recording. For example, Noor 3
stays outside calibration and fitting while Noor 1 and Noor 2 supply
training and validation. The final test model fits both complete training
recordings after its threshold is chosen.

With recording-level calibration, each inner model trains on one complete
Noor recording and scores the other. With blocked calibration, each training
recording is divided into four chronological quarters. Each inner model
trains on three quarters of both recordings and scores the remaining
quarter of each. The validation quarter rotates so every reviewed blink in
the two recordings supplies one calibration label. Fitting windows cannot
touch the validation blocks and there is a 150 ms guard outside each block.

The extra `rana1` windows go only into fitting in the auxiliary condition.
Model settings and the three outer test recordings remain the same.

Run from the repository root:

```sh
python3 -m src.model.run_continuous_pilot --max-clipped-fraction 0.25 --localize-peak --models svm_cnn_ensemble --calibration-split blocked --output reports/blink_calibration/noor_only
python3 -m src.model.run_continuous_pilot --max-clipped-fraction 0.25 --localize-peak --models svm_cnn_ensemble --calibration-split blocked --auxiliary-windows data/model/rana1 --output reports/blink_calibration/pooled
```

The blocked calibration metrics record the exact time ranges and usable
window counts. These tests estimate transfer between Noor recordings.
An independent test on a second person is a separate evaluation.

## Results

| Calibration | Threshold policy | Precision | Recall | F1 | False alarms/min |
| --- | --- | ---: | ---: | ---: | ---: |
| One recording at a time | f1 | 37.6% | 72.5% | 0.496 | 8.55 |
| One recording at a time | precision | 44.7% | 64.2% | 0.527 | 5.66 |
| 75% of both recordings | f1 | 46.6% | 62.9% | 0.535 | 5.13 |
| 75% of both recordings | precision | 65.9% | 37.1% | 0.475 | 1.37 |
| 75% of both + external training data | f1 | 43.4% | 57.6% | 0.495 | 5.35 |
| 75% of both + external training data | precision | 55.7% | 38.4% | 0.455 | 2.18 |

The blocked Noor-only pilot gives the best observed F1 of 0.535 and
catches 144 of 229 blinks, with 165 false alerts. The previous
precision-policy baseline caught 147 with 182 false alerts. The
more conservative blocked operating point improves precision and
catches fewer blinks. Adding the external package did not improve
the best F1 under this split.
