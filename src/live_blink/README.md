# BlinkIt live EEG blink detector

Connect BITalino and detect blinks with the saved `cnn_01` model. This is an EEG-only desktop app; it does not need the webcam. Two trained CNN seeds are averaged, exactly as in the reported CNN experiment. The user chose this CNN rather than the validation-selected CNN/XGBoost combination.

## Windows (the working BITalino laptop)

1. Extract `BlinkIt_Live_CNN.zip` completely to a local folder.
2. Install 64-bit Python 3.12 with the Python Launcher and Tcl/Tk, if not installed.
3. Open `src/live_blink/Setup_Windows.cmd` once. Internet is needed for dependencies. Model weights are already included.
4. Pair BITalino in Windows Bluetooth settings. Close OpenSignals and the dataset recorder; only one program can hold the device.
5. Open `src/live_blink/Run_Live_Blinks.cmd`.
6. Choose the board's outgoing COM port and the physically connected EEG channel (A4 for the training setup). Click **Connect & detect**.
7. Wait about four seconds for warm-up. Blink: the indicator flashes, a vertical orange line marks the estimated blink time, the counter increments, and a tone plays. **Test sound** checks audio; uncheck **Blink sound** to mute.
8. Click **Stop** before disconnecting or closing the board.

The app does not repair the earlier Mac Bluetooth problem. Use the Windows machine on which the board already streams successfully.

## This Mac checkout

Run `src/live_blink/Run_Live_Blinks.command`. It uses the prepared `.venv/rana-training` environment. No XGBoost, webcam model, OpenCV or PyAV is imported by this app.

## What is displayed

- Live filtered EEG, last ten seconds, in ADC counts (not calibrated microvolts).
- CNN score, with the fixed validation threshold of 0.80. Scores are not calibrated probabilities.
- Blink flash, count, sound and markers at estimated blink times.
- Warm-up, clipping, flat signal, missing packet and disconnected/no-data states.
- Current save directory. Live and replay logs are kept in separate directories.

**Optional replay** lets you select an existing recorder `eeg.csv`. It is visibly labeled **REPLAY — NOT LIVE**, does not connect Bluetooth, and does not load webcam labels to generate detections. Real BITalino input is the default. Replay is for software checks, not evidence of live hardware operation.

## Timing and model compatibility

BITalino streams 1,000Hz ADC from one A1–A4 channel. Filtering is the same third-order causal 0.5–12Hz Butterworth filter used for the CNN's training waveform. Each 1.5-second window is interpolated to 192 points, centered, normalized with the saved training scale and per-window scale, then scored by both checkpoints. Windows advance every 100ms. The filter remains continuous in live use; it resets at connection or detected sample gaps, not every minute as in the experimental split. A 3.5-second recovery buffer follows reset.

The model labels the window center, requiring ~750ms of future samples. Online peak confirmation adds 300ms: **about 1.05 seconds algorithm delay**, plus Bluetooth delivery, processing and UI refresh. This is delayed real-time detection, not an instantaneous blink response.

Offline evaluation used whole-block score suppression. The live app uses a bounded 300ms local maximum and a 350ms alert refractory interval. Therefore the offline 81.4% window accuracy / 76.2% event F1 is **not a measured live-app accuracy**. Continuous filtering and online alert behavior need prospective testing. Training labels and synchronization remain provisional. Sustained closure is not a separately trained class and motion can produce false detections. The app is for experimentation.

The 4-bit sample counter identifies many packet gaps but cannot detect missing multiples of 16 samples. Flat ADC windows suppress alerts; clipping is warned about but retained, matching training. A loss of input cannot generate new scores. No alerts are flushed from incomplete windows when stopping.

## Saved files

`data/raw/live_blinks/live/<timestamp_id>/` or `.../replay/<timestamp_id>/`:

- `eeg.csv`: relative sample index, packet sequence, raw ADC, host batch-receipt time.
- `scores.csv`: center time, model score, clipping fraction, flat flag.
- `blinks.csv`: estimated blink time, score, emission sample time, algorithm delay and host alert time.
- `session.json`: live/replay provenance, port/channel, model, completion/error status.

Times are relative to the first received EEG sample. They are not hardware-synchronized to an external camera. Files are written incrementally; Stop closes them. No data is uploaded.

## Verification

From the repo: `.venv/rana-training/bin/python -m unittest discover -s src/live_blink -p 'test_*.py' -v`.

Checks compare saved-window scores with original CNN predictions, streamed filter/window output with offline computation, arbitrary batch boundaries, gap recovery, duplicate suppression and delay, plus a 15-second real-recording replay. These checks do not verify physical Bluetooth, audible output or Windows GUI behavior. Use Test sound and a short live trial on the Windows laptop.
