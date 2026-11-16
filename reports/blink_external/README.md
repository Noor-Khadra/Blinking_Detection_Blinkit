# BlinkIt auxiliary-data comparison

The local [rana1 package](../../data/model/rana1/README.md) contains one public
EEGEyeNet participant and stays separate from the canonical Noor model data.
The experiments compare adding 28 or 56 external blink windows with a
Noor-only rerun. The training sampler supplies three no-blink windows for
each blink window within each training recording. The gain-matched run
scales the auxiliary waveforms to the median blink amplitude of the
current Noor training recordings.

Every evaluation uses the same three held-out Noor recordings, totaling
32.1 reviewed minutes and 229 blinks. The source participates in training
only. Each operating point is selected by validation on the Noor training
recordings. The rerun reproduced the previously reported Noor-only ensemble
and waveform SVM results exactly.

## Precision-oriented operating points

| Model | Training | Precision | Recall | F1 | False alarms/min |
| --- | --- | ---: | ---: | ---: | ---: |
| Ensemble | [Noor only](noor_only/metrics.json) | 44.7% | 64.2% | 0.527 | 5.66 |
| Ensemble | [Noor + 28 external blink windows](half/metrics.json) | 51.0% | 45.9% | 0.483 | 3.14 |
| Ensemble | [Noor + 56 external blink windows](pooled/metrics.json) | 46.0% | 52.4% | 0.490 | 4.39 |
| Ensemble | [Noor + 56, gain matched](gain_matched/metrics.json) | 49.3% | 48.9% | 0.491 | 3.58 |
| Normalized CNN | [Noor only](noor_only/metrics.json) | 36.8% | 43.7% | 0.399 | 5.35 |
| Normalized CNN | [Noor + 28 external blink windows](half/metrics.json) | 51.2% | 27.5% | 0.358 | 1.87 |
| Normalized CNN | [Noor + 56 external blink windows](pooled/metrics.json) | 55.4% | 33.6% | 0.418 | 1.93 |
| Waveform SVM | [Noor only](noor_only/metrics.json) | 34.5% | 54.6% | 0.423 | 7.37 |
| Waveform SVM | [Noor + 28 external blink windows](half/metrics.json) | 34.7% | 59.0% | 0.437 | 7.90 |
| Waveform SVM | [Noor + 56 external blink windows](pooled/metrics.json) | 31.7% | 62.0% | 0.419 | 9.52 |
| Waveform SVM | [Noor + 56, gain matched](gain_matched/metrics.json) | 28.8% | 61.6% | 0.393 | 10.82 |

The **Noor-only ensemble remains the best F1 candidate**. Direct pooling,
half pooling and gain matching all raised its precision and reduced recall.
The normalized CNN responded most strongly to the full auxiliary package.
It made 62 false alarms instead of 172, and detected 77 blinks instead of
100 at its precision-oriented setting. Keep that variant as an experimental
high-precision option. At the F1-oriented threshold the same CNN gave
40.8% precision, 56.3% recall, and 0.473 F1, compared with 35.8%, 66.4%,
and 0.466 for its Noor-only counterpart.

The [full comparison](comparison.csv) includes both operating points,
counts and false alarms per minute for every tested model and condition.
Each condition folder also has the thresholds, recording-level results
and alert records. These are exploratory results on one imported participant
and the same Noor development recordings. There is no independent
new-person result.

## Reproduce

Run from the repository root with the model dependencies installed:

```sh
python3 -m src.model.run_continuous_pilot --max-clipped-fraction 0.25 --localize-peak --models waveform_svm svm_cnn_ensemble cnn_polarity_normalized --output reports/blink_external/noor_only
python3 -m src.model.run_continuous_pilot --max-clipped-fraction 0.25 --localize-peak --models waveform_svm svm_cnn_ensemble cnn_polarity_normalized --auxiliary-windows data/model/rana1 --output reports/blink_external/pooled
python3 -m src.model.run_continuous_pilot --max-clipped-fraction 0.25 --localize-peak --models waveform_svm svm_cnn_ensemble cnn_polarity_normalized --auxiliary-windows data/model/rana1 --auxiliary-fraction 0.5 --output reports/blink_external/half
python3 -m src.model.run_continuous_pilot --max-clipped-fraction 0.25 --localize-peak --models waveform_svm svm_cnn_ensemble --auxiliary-windows data/model/rana1 --match-auxiliary-gain --output reports/blink_external/gain_matched
```

## Saved experimental candidate

The [precision-focused CNN](precision_candidate/detector.json) was refit on
all Noor recordings with the full `rana1` auxiliary package. Its thresholds
come from out-of-fold Noor predictions. It is saved alongside the comparison
for a future independent test.

```sh
python3 -m src.model.run_continuous_pilot --max-clipped-fraction 0.25 --localize-peak --auxiliary-windows data/model/rana1 --freeze-model cnn_polarity_normalized --output reports/blink_external/precision_candidate
```
