# Cross-person EEG blink transfer

The participant behind the `rana1` import is EEGEyeNet sub-EP10, while Noor's
recordings come from BlinkIt's BITalino setup. The forward and reverse
tests therefore change both participant and acquisition setup.

Frozen Noor models use their saved weights and thresholds. Training and
calibration of the reverse model use only the external recording, in
chronological blocks. Scoring on the external recording uses the eye-tracker
coverage intervals and annotated blink spans. Independently labeled intervals
define blink matches. Precision is undefined for a run with no alerts.

| Training → test | Operating point | True alerts | False alerts | Precision | Recall | F1 |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| Noor → external | boosted/f1 | 2 | 5 | 28.6% | 2.9% | 0.053 |
| Noor → external | boosted/precision | 1 | 0 | 100.0% | 1.5% | 0.029 |
| Noor → external | ensemble/f1 | 0 | 0 | — | 0.0% | 0.000 |
| Noor → external | ensemble/precision | 0 | 0 | — | 0.0% | 0.000 |
| External → Noor | boosted/f1 | 156 | 587 | 21.0% | 68.1% | 0.321 |
| External → Noor | boosted/precision | 179 | 822 | 17.9% | 78.2% | 0.291 |

## Blind Rahma run

A boosted model was fitted on Noor 1, Noor 2 and the external `rana1`
windows. The threshold and alert settings were chosen using blocked Noor
validation. It issued **141 balanced alerts** and
**85 precision-oriented alerts** on
Rahma's EEG. Rahma's labels were unused for model fitting or calibration.
Her review package has no matching capture timestamp file and the available
label alignment is provisional. The alert counts have no verified
precision or recall.

The saved [Rahma candidate](noor12_rana1_to_rahma/detector.json) and
[balanced alerts](noor12_rana1_to_rahma/f1_alerts.csv) can be scored after
independently verifying the labels and alignment. The complete results
and thresholds are in [metrics.json](metrics.json).

Run locally with `python3 -m src.model.run_blink_transfer`.
