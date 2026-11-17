# Rana + Noor 2 replay candidate

This is the highest-scoring candidate on our review-adjusted Rana test blocks. The [BITalino desktop demo](../../src/live_blink/README.md) loads the Rana-only CNN. The results on this page come from replay of saved EEG.

## Result on Rana

| Setting | Blink labels matched | Unmatched alerts | Precision | Recall | F1 | Algorithmic delay |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Existing Rana + Noor 2 alert rule | 117/159 | 14 | 89.3% | 73.6% | 80.7% | 1.05 s |
| Faster Rana + Noor 2 alert rule | **118/159** | **14** | **89.4%** | **74.2%** | **81.1%** | **0.95 s** |

The faster rule uses the same weights, a score threshold of 0.825, a local peak check over two score steps on each side (200 ms), and at least 350 ms between alerts. The original validation blocks selected the faster setting under the original rule's validation false-alert count. The one additional match is too small to establish a recall improvement. The 100 ms reduction is algorithmic and excludes Bluetooth, operating-system, and audio latency.

The review covered 59 disagreements between webcam labels and detector alerts, and the original matched cases retain their automatic labels. One confirmed alert occurred close to another reviewed blink, so the 159-label total is provisional. Test blocks have been used in prior development comparisons. The faster setting has not been tested with live BITalino input. These results apply to the examined Rana recordings.

## Checkpoints and scoring

The two CNN checkpoints were fine-tuned from the matching Rana-only seeds using training blocks from `rana_01_clean` and `rana_02` plus reviewed examples from the earlier Noor 2 recording. The model uses a causal 0.5–12 Hz filter at 1 kHz, a centered 1.5-second window, and emits a score every 100 ms. Resample each window to 192 points, center it, and apply the two normalization views used by [the original inference engine](../../src/live_blink/engine.py), using each checkpoint's own `scale`. Average the sigmoid scores from both checkpoints, then apply the settings in the manifest. The saved replay report uses one-to-one matching within 250 ms over intervals with usable webcam coverage.

[Checkpoint hashes and exact settings](manifest.json) · [Validation choices](existing_gate_validation_choices.json) · [Replay metrics](existing_gate_test.csv)
