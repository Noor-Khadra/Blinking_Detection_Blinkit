# BlinkIt · EEG blink detection

A single BITalino EEG channel drives the live blink marker, counter and tone. The webcam provided reference labels during data collection. The deployed detector averages two CNNs trained on Rana 01 and Rana 02. Noor 4 is the second-wearer transfer experiment.

[Project report](submission/BlinkIt_Report.pdf) · [Presentation](submission/BlinkIt_Presentation.pptx) · [Live detector](src/live_blink/README.md) · [Recorder](src/acquisition/README.md)

## Run the demo

On Windows, install 64-bit Python 3.12 with the Python Launcher and Tcl/Tk. Clone this repository or extract its ZIP. Open `src/live_blink/Setup_Windows.cmd` once, then `Run_Live_Blinks.cmd`. Pair the BITalino, close any other acquisition app and select the outgoing COM port and EEG input A4. For a hardware-free demonstration choose replay and open `data/raw/datasets/rana_01_clean/eeg.csv`. The saved weights and scaling values are in [models](src/live_blink/models).

## What was evaluated

| Question | Training and testing | Evidence |
|---|---|---|
| Which feature baseline works? | Logistic regression, RBF SVM, Extra Trees and gradient boosting on Rana time blocks | [Baseline comparison](reports/rana_new_training/README.md) |
| Does an external settings search help? | Hyperparameters explored on public EEGEyeNet EP10, candidate models fitted on Rana | [Tuning results](reports/rana_public_hyperparameters/README.md) |
| Does learning a blink waveform help? | 12 XGBoost settings, four CNN configurations with two seeds each, and three CNN/XGBoost blending weights | [Offline model comparison](reports/rana_cnn_xgb/README.md) |
| Does it transfer to a different wearer? | Frozen Rana → Noor 4, pooled Rana + Noor 4 on held-out minutes, and Noor 4 → Rana | [Transfer results](reports/rana_noor4_cnn/README.md) |
| Can we reduce false alerts in continuous use? | Deployed rule, signal-quality gates, difficult negative examples, wider filter, clipping exclusions and score blending | [Streaming settings and scores](reports/live_blink_experiments/comparison.csv) |
| Does window length matter on a new session? | Train and validate on one Rana session and test the other, in both directions. Compare 0.75, 1.0 and 1.5 seconds and the shorter-window blend | [Session-swap results](reports/rana_window_length/summary.csv) |

For the two Rana sessions, one-based minutes 1, 2, 4, 6, 7 and 9 supply training. Minutes 3 and 8 supply validation. Minutes 5 and 10 supply testing. Boundary guards and tracking checks limit usable time. The later recordings were unscripted. The guided ten-block recorder shown in the presentation is the repeat protocol.

## Data and interpretation

Raw numerical recordings, capture timestamps and webcam labels are in [data/raw/datasets](data/raw/datasets). Aligned events, example indices and split manifests are in [data/model](data/model). The [recording release](https://github.com/Noor-Khadra/Blinking_Detection_Blinkit/releases/tag/blinkit-recordings-20261116) contains the original three archives including videos. Verify and restore with `python src/acquisition/restore_recordings.py PATH_TO_ARCHIVES`. Public tuning inputs and their attribution are in [data/external/eegeyenet_ep10](data/external/eegeyenet_ep10).

The submitted live rule matched 112 of 158 reference blinks on Rana replay with 17 unmatched alerts (86.8% precision, 70.9% recall, 78.0% event F1). The frozen offline CNN scored 31.0% F1 across scorable Noor 4 intervals. These results use different evaluation rules. Automatic webcam labels, estimated host-clock alignment and repeated use of the same test minutes limit interpretation.

## Reproduce

Run from the repository root in a fresh Python virtual environment. The scripts write outputs under data/model and reports, so use a separate checkout when retraining. Windows live setup is separate from this analysis environment.

```sh
python -m pip install -r src/model/requirements-rana-cnn-xgb.txt
python src/model/train_rana_new.py
python src/model/tune_public_transfer.py
python src/model/train_rana_cnn_xgb.py
python src/model/train_noor4_addition.py
python src/model/experiment_live_blink.py
python src/model/experiment_rana_window_length.py
python -m unittest discover -s src/live_blink -p "test_*.py" -v
```

On macOS the two training launchers in src/model expect .venv/rana-training and configure PyTorch's bundled OpenMP library for XGBoost. See the [streaming README](reports/live_blink_experiments/README.md) for the corresponding shell environment when running the experiments directly.
