# Webcam processing and labeling

Rana's scope is the webcam pipeline and reviewed video labels. EEG alignment,
EEG preprocessing, training, and live inference belong to the other stages.

## Run

Use Python 3.11–3.13 in a virtual environment. Install the requirements in this
directory. PyAV decodes frames with their source timestamps; no separate FFmpeg installation is needed. Obtain the official
[Face Landmarker model](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task)
and save it as `face_landmarker.task`. The Tasks Face Landmarker API supplies
the eye landmarks used by the plan's Face Mesh EAR method.

From the repository root:

```sh
python -m pip install -r src/webcam/requirements.txt
python src/webcam/extract_landmarks.py data/raw/noor/session1/webcam.mp4 face_landmarker.task data/derived/noor/session1/webcam_ear.csv
python src/webcam/build_candidates.py data/derived/noor/session1/webcam_ear.csv data/derived/noor/session1 --eye mean --threshold-ratio 0.55
```

Choose `--eye left` or `--eye right` to match the participant's EEG electrode
side; mean is an explicit default when that side is unknown. No silent fallback
is applied to missing selected-eye measurements. Use `--open-ear VALUE` for an
open-eye reference measured from verified footage. Otherwise the smoothed
session EAR's 80th percentile estimates the reference; it is not calibration
from a verified eyes-open segment. Tune on the pilot, then record the settings.

## Method and files

### Live laptop webcam trial

On macOS, double-click `Run_Live_Webcam.command`, or run it from the repository root:

```sh
src/webcam/Run_Live_Webcam.command
```

The launcher uses `.venv/live-webcam`. To set it up on another checkout, run
`python3 -m venv .venv/live-webcam` and then
`.venv/live-webcam/bin/python -m pip install -r src/webcam/requirements.txt`.
Put the Face Landmarker model linked above in the repository root. If macOS
prompts for camera access, allow it for the terminal running the launcher.
Use `--camera 1` if the default camera index selects the wrong device.

The mirrored preview shows live face landmarks, eye contours, and both eyes'
EAR. With the preview window focused, press **R** to start or stop recording
and **Q** to quit. For a quick trial, look straight ahead, blink a few times,
then hold your eyes closed briefly and reopen them.

Each trial saves a clean `webcam.mp4` and a frame-by-frame
`webcam_timestamps.csv` under `data/raw/live_webcam/`. Capture times are
measured when the app receives a frame. The CSV includes elapsed time,
monotonic time, UTC, and MediaPipe eye measurements. It does not measure
the camera's hardware exposure time. The preview overlay is drawn on a copy.

### Offline extraction

Extraction uses every decoded frame, source presentation timestamps, one face,
pixel-coordinate EAR for each eye, mean EAR, and optional blink blendshape
scores. Missing face frames remain missing. Blendshape scores are not blink
probabilities. A frame-count mismatch or invalid time sequence raises an error.

Candidate generation applies a three-frame median without crossing missing
tracking or timestamp gaps of 150 ms or more. It finds selected-eye EAR below
the participant reference times the threshold ratio. Peak is minimum smoothed
EAR; offset is the first tracked frame above threshold when available. Very
short and over-0.5-second intervals are flagged instead of silently discarded.
This deliberate difference from the draft plan retains events for review.

- `webcam_ear.csv`: zero-based frame index, encoded video time in seconds,
  both-eye and mean EAR, face detection, blink scores and diagnostic geometry.
- `blink_candidates_UNREVIEWED.csv`: proposed onset, peak, offset, frame indices,
  minimum smoothed EAR, event type, review status, flags, notes.
- `processing_summary.json`: chosen eye, reference, threshold and missing count.
- `review_data.json` and `Review_blinks.html`: standalone review data and player.

## Review and handoff

Open `Review_blinks.html` and choose the exact video used for extraction.
Check the start/end sync blinks, deliberate blinks, gaze changes and head turns.
Watch the complete recording to add missed events. Reject merged candidates
and add individual blinks where needed. Classify winks and sustained closure
separately. Visually adjust onset to first closing frame, peak to greatest
closure, offset to first clearly reopened frame. These visible boundaries can
differ from the automatic threshold crossings.

Export `labels.csv` and save `review_state.json` regularly. Browser storage is
only a convenience. Keep each session's review page in its own directory.
`minimum_ear` describes the original proposal and is not recomputed after manual
boundary edits; manually added events leave it blank. The Accept blink button requires a reviewer name. Use rows with
`review_status=accepted`, `event_type=blink`, and `human_review_status=completed`
for human-reviewed blink labels. AI draft assessments alone never satisfy this gate.
For the Noor pilot, Rana completed the human review and explicitly waived an additional reviewer. The legacy second_review_status column in her exports is not a requirement gate.

Hand Person C the verified CSV, EAR CSV, summary, exact source video identity,
and acquisition timestamps/session metadata. All generated times are encoded
video playback times, not EEG or monotonic capture times. This matters for
constant-rate AVI recordings whose real capture pace differs from playback.
Person C must establish alignment and drift separately. The ±150 ms matching
tolerance is not evidence that synchronization is correct. At roughly 30 fps,
visual boundaries have roughly one-frame resolution, not millisecond accuracy.

The upload CSVs for sessions 1–3 contain accepted blinks and sustained closures.
Full review exports and detailed eye measurements are retained in the project
working files. See [labeling_runs](labeling_runs/) for the scripts used to
produce the reviewed recordings' candidate proposals.

## Validation

```sh
python -m unittest discover -s tests -v
```

Tests cover reopened-frame offsets, missing-tracking boundaries and invalid
timestamps. Actual eyelid tracking accuracy still requires video review.
