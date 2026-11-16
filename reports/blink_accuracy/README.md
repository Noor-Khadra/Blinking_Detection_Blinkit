# BlinkIt accuracy experiments

The experiments use the same 229 reviewed blinks and 32.1 minutes of
continuous EEG from Noor 1–3. Each outer fold holds out one complete
recording. The four chronological calibration folds use the remaining
recordings to choose thresholds and alert settings. The alert search covers
window-center timestamps, nearby EEG peaks, optional score smoothing, and
240, 350, and 500 ms suppression distances. The held-out recordings stay
outside fitting and parameter selection for each fold.

## Most useful changes

| Configuration | Precision | Recall | F1 | False alarms/min |
| --- | ---: | ---: | ---: | ---: |
| Previous blocked-calibration ensemble | 46.6% | 62.9% | 0.535 | 5.13 |
| Ensemble with alert settings selected on validation | 48.2% | 59.8% | 0.534 | 4.57 |
| Expanded waveform SVM | 51.8% | 57.6% | 0.545 | 3.83 |
| Boosted waveform model | 52.2% | 62.0% | 0.567 | 4.04 |
| Boosted model with difficult negatives, 25% clipping gate | 57.3% | 59.8% | 0.585 | 3.17 |
| Center-focused CNN | 66.0% | 54.1% | 0.595 | 1.99 |
| Fusion of boosted model and center-focused CNN | 66.0% | 60.3% | 0.630 | 2.21 |
| **Boosted model with difficult negatives, 45% clipping gate** | **64.0%** | **66.8%** | **0.654** | **2.68** |
| Same boosted model, precision-oriented threshold | 78.3% | 39.3% | 0.523 | 0.78 |

The selected boosted model uses waveform shape, local signal-to-background
ratio, and spectral features. Its negatives include open-eye windows that a
preliminary classifier scores highly and verified sustained-closure windows.
The source for mining remains within each training fold. At the same recall
as the ensemble with alert settings selected on validation, this training
change decreased false alerts from 147 to 102 at the 25% gate.

The clipping gate is the maximum clipped fraction allowed within a window.
Retesting the boosted model at 25%, 35%, 45%, and 60% gave balanced F1 of
0.585, 0.634, 0.654, and 0.597. The 45% gate achieved F1 of **0.638–0.654**
across three training seeds. The precision profile varied more, catching
31–50% of reviewed blinks with 78–80% alert precision across seeds.

At the balanced operating point, the selected seed caught 48/62 blinks on
Noor 1, 59/81 on Noor 2, and 46/86 on Noor 3. Compared with the previous
blocked-calibration ensemble, this is nine more detected blinks and 79
fewer false alarms. Noor 3 still generated 44 of the 86 false alarms.
These runs reuse Noor recordings for development. Performance on a
different person and real-time latency remain unmeasured.

## Files

- [All pooled results](comparison.csv)
- [Waveform and mining runs](shape25/metrics.json)
- [Neural model and fusion runs](neural25/metrics.json)
- Clipping comparisons at [35%](clipping35/metrics.json), [45%](clipping45/metrics.json), and [60%](clipping60/metrics.json)
- Seed checks at [25%](seed_check25/metrics.json) and [45%](seed_check45/metrics.json)
- [Candidate configuration](candidate/detector.json) and [weights](candidate/model.joblib)

The candidate refits on all three Noor recordings and saves calibration
settings for a future independent test.

## Reproduce

```sh
python3 -m src.model.run_blink_accuracy --models prior shape_svm boost boost_mined --output reports/blink_accuracy/shape25
python3 -m src.model.run_blink_accuracy --models center_cnn fusion --output reports/blink_accuracy/neural25
python3 -m src.model.run_blink_accuracy --quality .45 --models prior boost_mined --output reports/blink_accuracy/clipping45
python3 -m src.model.run_blink_accuracy --quality .35 --models boost_mined --output reports/blink_accuracy/clipping35
python3 -m src.model.run_blink_accuracy --quality .60 --models boost_mined --output reports/blink_accuracy/clipping60
python3 -m src.model.run_blink_accuracy --models boost boost_mined --seeds 17 73 --output reports/blink_accuracy/seed_check25
python3 -m src.model.run_blink_accuracy --quality .45 --models boost_mined --seeds 17 73 --output reports/blink_accuracy/seed_check45
python3 -m src.model.run_blink_accuracy --quality .45 --freeze boost_mined --output reports/blink_accuracy/candidate
```
