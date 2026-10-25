# Webcam pilot validation — 25 October 2026

Branch: `webcam-mediapipe`. Scope: webcam processing and label preparation.

## Extraction executed

MediaPipe 1.1.1 Face Landmarker ran in VIDEO mode on the available compressed
Noor review video. The original large recording was not available locally.
Decoded and container-declared frame counts both equal 20,520. Every decoded
frame received face landmarks; this does not measure eyelid tracking accuracy.
Presentation timestamps are strictly increasing, with no MediaPipe timestamp
adjustments. Last frame: 675.632133 seconds. Extraction runtime: 177.6 seconds.

Input SHA-256: `e421535f4bd99b8059bdcd1dce6f7d6a8776ca1b8f0631fa56bf49968509bd41`.
Model SHA-256: `64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff`.
The per-frame timestamp array matches the supplied original-extraction CSV
exactly. EAR correlation with that CSV: left 0.99567, right 0.99622; mean absolute
EAR differences: left 0.00622, right 0.00583. These are consistency checks, not
accuracy against manually labeled eyelids.

## Pilot tuning and visual screening

Mean-eye EAR with a three-frame median and an estimated open reference of
0.35647 was used. Threshold ratio 0.55 produced 73 unreviewed proposals in the
new extraction. The earlier 0.70 run on supplied measurements produced 105.
Visual inspection found open eyes during several low-EAR gaze intervals;
a stricter ratio reduces these proposals. A 0.45 exploratory ratio omitted
some visibly closing events, so 0.55 is the provisional pilot setting.
This is exploratory tuning, not held-out detector evaluation.

The complete timeline was screened using 6,840 eye crops (every third frame,
roughly 10 fps) arranged in 35 timestamped sheets. All 105 earlier proposals
were also inspected in nine-frame filmstrips at two-frame stride around their
peaks. This is AI visual screening; it is not continuous video viewing or a
human frame-by-frame boundary annotation. Very brief events can be missed.

First-pass assessments of those 105 proposals: 40 probable blinks, 12 probable
sustained closures, 8 ambiguous closures, and 45 probable gaze/occlusion events.
These are assessment categories, not verified event counts. The final review
queue contains the 73 new proposals plus two visually flagged short events
from the earlier run. Its 75 rows include 40 probable blinks, 13 sustained
closures, 8 ambiguous closures, and 14 likely false proposals retained for audit.
Different counts result from different proposal boundaries and splits.

The long closed-eye footage around 603–637 and 640–669 seconds should not be
counted as a series of spontaneous blinks merely because tracking thresholds
split it into several intervals. Intent/activity cannot be established from
these eye crops alone. A three-blink synchronization pattern was not confirmed
at the start and end; do not assume the draft acquisition protocol was followed.

## Outputs and remaining review

Session outputs are in `data/derived/noor/session1/` (ignored by Git):

- `webcam_ear.csv` and `.extraction.json`: new per-frame extraction and provenance.
- `blink_candidates_UNREVIEWED.csv`: 73 automatic proposals.
- `candidate_audit_AI_DRAFT.csv`: visual screening notes for all 105 earlier proposals.
- `labels_AI_REVIEW_DRAFT.csv`: 75-row queue; no rows are accepted as training labels.
- `Review_AI_DRAFT.html`, `video.mp4`, and `review_state_AI_DRAFT.json`: portable review.

Open `Review_AI_DRAFT.html`. Review each uncertain event, confirm blink versus
wink/partial closure, set visible onset/peak/reopening at full frame resolution,
and enter the reviewer name before accepting a blink. Check between proposals
for misses; the sparse AI screen is not proof of no blinks in the gaps.
Save CSV and JSON. Ask a second human reviewer to spot-check positives,
rejections, boundaries and a sample of intervening footage, recording their
name, intervals checked, discrepancies and resolutions in notes. The independent
human review has NOT been performed. No external person has been contacted.

Only accepted blink rows with `human_review_status=completed` are eligible for
the modeling handoff after that quality check. EEG alignment remains Person C's
stage. No EEG-clock labels, verified labels.csv, recall/precision metrics, or
independent reviewer sign-off are claimed.

## Software checks

Three unit tests pass: reopened-frame offsets, missing tracking splits, and
invalid timestamps. Both Python scripts compile. Generated review JavaScript
passes Node syntax checking. `pip check` reports no broken requirements. The
MediaPipe extraction ran end to end. Browser interaction has not been tested.
