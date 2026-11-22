# BlinkIt dataset recorder

## Windows quick start

1. Extract the ZIP to a local folder. Install **64-bit Python 3.12** from
   [python.org](https://www.python.org/downloads/windows/) with the Python
   Launcher and Tcl/Tk options enabled. In `src/acquisition`, double-click
   `Setup_Windows.cmd` once. It installs the dependencies and downloads the
   official Face Landmarker model.
2. Turn on the BITalino and pair it under **Settings → Bluetooth & devices**.
   Enter `1234` if asked. Find the board's **outgoing** COM port under **More
   Bluetooth settings → COM Ports**, for example `COM7`. Close OpenSignals
   before running BlinkIt.
3. Double-click `Run_Dataset_Recorder.cmd`. Select the outgoing port and the
   channel the EEG sensor is physically connected to. Click **Connect
   devices**. When the EEG and EAR traces are live, record a short test and
   use **Stop early** before attempting a full trial.

For a camera-off connection check, open Command Prompt in the extracted
folder and run the following, substituting the actual port and channel:

```bat
src\acquisition\Run_Dataset_Recorder.cmd --test-eeg --port COM7 --channel 4
```

## macOS quick start

On this Mac, run the launcher from Terminal or double-click
`Run_Dataset_Recorder.command` in Finder. The camera permission belongs to
the terminal running the app. Close the browser webcam preview and release the
BITalino connection in OpenSignals before connecting in the recorder.

```sh
src/acquisition/Run_Dataset_Recorder.command
```

1. Pair the BITalino and select its serial port (`/dev/cu.…`) in the app.
   You can paste a path or supply `--port` at launch. Choose the EEG input
   you physically used. Select A4 for the presented setup. Sampling is at 1,000 Hz.
2. Click **Connect devices**. Both devices stream in preview before they are
   ready. Enter a new session name such as `participant03_session02`.
3. Click **Start recording** once. Both streams use the same host-clock start
   and stop boundaries. The webcam landmarks, raw EEG graph and left/right
   EAR graph update during the recording. Both graphs show the last ten
   seconds on the same estimated host timeline. The EEG graph shows the
   selected channel in raw ADC counts. Use **Stop early** to save a partial session. Full sessions stop
   automatically and preserve the hardware connections for the next name.

## Ten-minute protocol

Each run takes 12 minutes 30 seconds and saves ten one-minute activities. Before each activity, allow 12 seconds to read and 3 seconds for the countdown. Only activity frames and EEG samples are saved. EEG sample indices and session times retain the gaps. Video playback skips the pauses. For modeling, use session time and build windows separately within each block. The brighter start tone signals when to act. Keep the electrodes in place for all three runs. Use a fresh session name each time.

Choose order A, B or C for a custom session name. The guided repeat protocol is shown as a design in the presentation. The three evaluated recordings use the earlier unscripted acquisition procedure.

A full-screen participant view contains the reading passage, spoken prompts as on-screen text, conversation questions, and an offline 60-second harbor animation. Head turns, facial actions and single, double and triple blinks appear one at a time. Use Test countdown sound before starting. A higher tone cues reopening after sustained closure. Escape leaves full-screen mode. Leave participant audio enabled.

Each run includes quiet fixation, reading, video, cued blinks, gaze shifts, head movement, talking, facial movements, sustained closures and a mixed activity. The order is saved in session.json. Protocol.csv lists planned boundaries. Cues.csv records prompts dispatched by the interface and their session times. Actual participant actions can differ from prompts. Webcam frames and exported events carry the block and phase. Use the timeline for extracting EEG windows later. Automatic blink and open-eye labels come from the webcam.

## Files

Each trial writes to `data/raw/datasets/<session-name>/`.

| File | Contents |
|---|---|
| `webcam.mp4` | Clean video, written in fragments during recording |
| `webcam_frames.csv` | Frame index, video time, shared session time, condition, phase, EAR, blink blendshape scores |
| `eeg.csv` | Raw ADC samples, original sample indices, packet counters and digital channels |
| `eeg_clock.csv` | Host-clock observations for fitting the EEG timeline |
| `protocol.csv`, `session.json` | Activity timing, acquisition settings, counts, and timing diagnostics |
| `eeg_timeline.npz` | ADC waveform, sample indices and aligned session times for Python |
| `eeg_events_auto.csv` | Automatic blink and sustained-closure labels with onset, peak, end, frame and EEG sample indices |
| `no_blink_coverage_auto.csv` | Automatically inferred open-eye intervals for sampling no-blink windows |
| `Review_blinks.html` | Optional video and EAR viewer |

Video times come from when the Mac receives webcam frames. The EEG timeline
uses device counters and host batch receipt times. The same host clock
eliminates the separate start-button offset. Camera delivery and Bluetooth
latency can still differ. Check a short trial against recognizable blinks or
a shared hardware marker before treating the alignment as validated. The
digital inputs and the original sample indices are preserved.

Model preparation can use the automatic labels directly. They retain their
MediaPipe/EAR provenance and unreviewed status. A complete session with
usable clock diagnostics marks mapped activity events as eligible. No-blink
coverage requires both eyes to have valid, above-threshold EAR and excludes
countdowns, the eyes-closed block, tracking gaps, and the area around detected
events. Automatic labels can still miss blinks or confuse artifacts with blinks.

To regenerate the automatic labels and coverage for a saved session:

```sh
src/acquisition/Run_Dataset_Recorder.command --auto-labels data/raw/datasets/participant03_session02
```

If you choose to review a session separately, use `--reviewed-labels SESSION_DIR CSV`
to map the exported human labels to EEG. Model-specific windows and dataset splits
are still built during training-data preparation.

## Setup for a fresh checkout

Use Python 3.11–3.13 with Tk support and place the official [Face Landmarker model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
at the repository root as `face_landmarker.task`.

```sh
python3 -m venv .venv/live-webcam
.venv/live-webcam/bin/python -m pip install -r src/webcam/requirements.txt pyserial==3.5 pillow
.venv/live-webcam/bin/python -m pip install --no-deps bitalino==1.2.6
```

The Mac path uses the paired serial port via pySerial. The optional PyBluez
dependency is outside this connection path.
