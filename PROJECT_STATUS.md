# BlinkIt experiment and live-demo status

## Active live model

**The live interface uses the Rana-only `cnn_01` model**, averaging seeds 20261116 and 20261117. Its checkpoints are in `src/live_blink/models/`, with provenance and hashes in `manifest.json`. The Rana+Noor model has NOT replaced it.

Windows: extract `dist/BlinkIt_Live_CNN.zip`, run `src/live_blink/Setup_Windows.cmd`, then `Run_Live_Blinks.cmd`. The interface plots EEG, marks detected blinks and plays a sound. The model has about 1.05 seconds of algorithm delay plus device delivery latency. See [live interface documentation](src/live_blink/README.md).

## Experiments

| Experiment | Report | Main finding |
|---|---|---|
| Four conventional models on two real Rana recordings | [Baseline](reports/rana_new_training/README.md) | Validation-selected SVM: 66.9% test event F1 |
| Public-data hyperparameter search; fresh Rana-only fits | [Public tuning](reports/rana_public_hyperparameters/README.md) | Selected SVM: 68.2% event F1; no public weights transferred |
| 1D CNN and actual XGBoost | [CNN/XGBoost](reports/rana_cnn_xgb/README.md) | CNN: 82.4% precision, 70.9% recall, 76.2% event F1 on reused test blocks |
| New Noor4 recording and pooled training | [Rana + Noor](reports/rana_noor4_cnn/README.md) | Pooled per-person event F1: Rana 77.6%, Noor 27.0%; not deployed |

CNN's offline window accuracy was 81.4%; this is not live event accuracy. The CNN+XGBoost combination won validation in that experiment, while the user chose the standalone CNN for the live demo. Test blocks have been reused across development experiments. Automatic webcam labels are unreviewed and timestamp alignment is provisional. No claim of millisecond-accurate hardware synchronization or generalization to arbitrary new people is made.

## New recording data

The real new recordings are `rana_01_clean`, `rana_02`, and `noor4-clean`. Raw EEG, frame measurements, event labels, clock files and prepared arrays are included in this snapshot. Original recording ZIPs, including the large webcam videos, are published as assets in the [recording release](https://github.com/Noor-Khadra/Blinking_Detection_Blinkit/releases/tag/blinkit-recordings-20261116). Checksums are in [data/recording_release.json](data/recording_release.json).

To restore the videos and complete original folders:

```sh
gh release download blinkit-recordings-20261116 --repo Noor-Khadra/Blinking_Detection_Blinkit --dir /tmp/blinkit-recordings
python3 src/acquisition/restore_recordings.py /tmp/blinkit-recordings
```

The historical `rana1` alias under `data/model/rana1` and `data/derived/rana/session1` is **external EEGEyeNet participant EP10**, not a recording of Rana. The provenance JSONs identify its source, synthetic frame fields and resampling. It is excluded from the new Rana/Noor CNN training. Older experiments remain for traceability.

## Reproduction and checks

- Model requirements: `src/model/requirements-rana-new.txt` and `requirements-rana-cnn-xgb.txt`.
- Rana preparation/baselines: `python3 src/model/train_rana_new.py`.
- CNN/XGBoost on this Mac: `src/model/Run_Rana_CNN_XGB.command`.
- Noor addition: `src/model/Run_Noor4_Training.command`.
- Live-app software tests: `python -m unittest discover -s src/live_blink -p 'test_*.py' -v`.

Five live-app software tests passed, including model score parity, streaming filter equivalence, gap recovery, alert delay and connection-error cleanup. Physical Windows Bluetooth, GUI display and audible output still require an on-device trial. The Mac's earlier Bluetooth connection problem is not fixed by this interface.
