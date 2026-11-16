# Continuous EEG blink detection experiments

## Evaluation

The test data are approximately 32 minutes of continuously reviewed EEG
with 229 labeled blinks across Noor 1–3. A test fold holds out one complete
recording. Every reviewed blink counts toward recall, including blinks that
were outside the first window benchmark or fell in a clipped EEG segment.
Sustained closures stay distinct and an alert during a sustained closure
counts as a false alarm. All recordings belong to Noor. Rahma's EEG
annotations still need independent confirmation.

Models scan one-second EEG windows at a 100 ms stride. Inputs are
median-centered, antialias-filtered and reduced to 100 Hz. Training includes
eligible windows at several positions around reviewed blink peaks and
samples three open-eye windows per blink window within each training
recording. The CNN resamples these negatives each epoch. The signal gate
skips non-contiguous, flat and strongly clipped windows. The comparison
tests allowing either 10% or 25% of a window to be at the ADC limits.
For Noor 3, the 25% gate admitted 66 positive training windows versus
18 with the 10% gate. Multiple windows may represent a single blink.

Two folds within the training recordings set each test threshold. The
precision policy tries to reach 80% validation precision with recall of at
least 20%, then falls back to the threshold with the best F0.5. F0.5 gives
precision more weight than recall. Each emitted alert can match at most one
annotated blink, with 150 ms of tolerance beyond either boundary. Suppression
keeps the highest-scoring alert within a 350 ms neighborhood. The
localization variant timestamps it at a nearby EEG deflection.

## Development results

| Configuration | Precision | Recall | F1 | False alarms/min |
| --- | ---: | ---: | ---: | ---: |
| Amplitude rule, 25% gate | 19.1% | 28.4% | 0.228 | 8.59 |
| Normalized CNN, 10% gate | 30.7% | 26.6% | 0.285 | 4.29 |
| Hard-negative CNN, 10% gate | 28.4% | 33.6% | 0.308 | 6.03 |
| Localized plain CNN, 25% gate | 45.9% | 46.7% | 0.463 | 3.92 |
| Localized waveform SVM, 25% gate | 34.5% | 54.6% | 0.423 | 7.37 |
| Localized SVM + normalized CNN, 25% gate | 44.7% | 64.2% | 0.527 | 5.66 |
| Same ensemble, recall-oriented operating point | 37.6% | 72.5% | 0.496 | 8.55 |

The ensemble combines the waveform SVM and polarity- and gain-augmented CNN
scores using their geometric mean. The CNN uses within-window noise
normalization. At the precision-oriented operating point the ensemble
detected 49/62 blinks on Noor 1, 57/81 on Noor 2 and 41/86 on Noor 3.
The plain CNN achieved the highest pooled precision, but caught only
one Noor 3 blink. The ensemble achieved the strongest observed F1 and
F0.5 and was selected as the development candidate.

These figures come from the same three development recordings used to
compare configurations. The ensemble fit on all Noor recordings is saved
for a future independent evaluation. Its thresholds were calibrated using
out-of-fold Noor predictions. These offline results use windows centered
on each scan time and provisional event timing, so they do not establish
real-time latency or performance on a new person.

## Reproduce

Run from the repository root. Install the dependencies in
`src/model/requirements.txt` and fetch `feat/eeg-preprocessing` if its raw
recordings are unavailable locally.

```sh
python3 -m src.model.run_continuous_pilot --models amplitude adaptive_amplitude waveform_svm cnn cnn_polarity cnn_polarity_normalized --output reports/blink_continuous/clip10
python3 -m src.model.run_continuous_pilot --max-clipped-fraction 0.25 --models amplitude adaptive_amplitude waveform_svm cnn cnn_polarity cnn_polarity_normalized --output reports/blink_continuous/clip25
python3 -m src.model.run_continuous_pilot --models cnn_hard_negative --output reports/blink_continuous/hard10
python3 -m src.model.run_continuous_pilot --max-clipped-fraction 0.25 --localize-peak --models cnn cnn_polarity_normalized --output reports/blink_continuous/localized25
python3 -m src.model.run_continuous_pilot --max-clipped-fraction 0.25 --localize-peak --models waveform_svm svm_cnn_ensemble --output reports/blink_continuous/localized_hybrid25
python3 -m src.model.run_continuous_pilot --max-clipped-fraction 0.25 --localize-peak --freeze-model svm_cnn_ensemble --output reports/blink_continuous/candidates/svm_cnn_ensemble
```
