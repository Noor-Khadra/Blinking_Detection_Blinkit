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
   you physically used. The app defaults to A3 and samples at 1,000 Hz.
2. Click **Connect devices**. Both devices stream in preview before they are
   ready. Enter a new session name such as `participant03_session02`.
3. Click **Start recording** once. Both streams use the same host-clock start
   and stop boundaries. The webcam landmarks, raw EEG graph and left/right
   EAR graph update during the recording. Both graphs show the last ten
   seconds on the same estimated host timeline. The EEG graph shows the
   selected channel in raw ADC counts. Use **Stop early** to save a partial session. Full sessions stop
   automatically and preserve the hardware connections for the next name.

## Ten-minute protocol

The [recording guidelines](https://docs.google.com/document/d/1wD1WGZYFQbeyb9kgjj-LlHo_UHn5Zugk/edit)
list 11 minutes of activities. The app scales those durations to fit ten
minutes including eight recorded three-second countdowns. The display shows
the next instruction during each countdown and sounds a system bell.

| Activity | Countdown begins | Activity time in seconds | Activity duration |
|---|---:|---:|---:|
| Fixed gaze, eyes open | 0 | [3, 82) | 79 s |
| Silent reading | 82 | [85, 164) | 79 s |
| Watch a video | 164 | [167, 246) | 79 s |
| Talk or read aloud | 246 | [249, 328) | 79 s |
| Voluntary blinks | 328 | [331, 435) | 104 s |
| Eye movements | 435 | [438, 490) | 52 s |
| Other artifacts | 490 | [493, 545) | 52 s |
| Rest with eyes closed | 545 | [548, 600) | 52 s |

Prepare the reading material and video before starting. The app shows
instructions for each activity. Eye movements and other artifacts can still
contain real blinks. Section and phase describe the task at that time.

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
