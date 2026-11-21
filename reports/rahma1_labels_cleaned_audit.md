# Label audit: `data/derived/noor/session4/rahma1_labels_cleaned.csv` (accepted rows)

| Check | Count |
|---|---:|
| Labels in file | 175 |
| Accepted blinks | 157 |
| Other event types | 18 |
| Near-duplicate labels (peaks < 100 ms apart) | 0 |
| Overlapping labels | 0 |
| Blinks longer than 1.5 s | 0 |
| Blinks 0.8-1.5 s | 1 |
| Boundary/frame problems | 0 |
| Labels needing a video check | 1 |
| Labels with no issue | 174 |

Blink duration (s): median 0.300, IQR 0.267-0.400, max 0.83.

Thresholds: duplicate < 100 ms between peaks; long blink > 0.8 s; closure > 1.5 s; frame tolerance 1.5 frames at 30 fps.

## Labels to check in the video

| event_id | onset_s | duration_s | issues | suggested action |
|---|---:|---:|---|---|
| 56 | 603.833 | 0.832 | long_blink_0.8-1.5s | check video: tighten offset or mark as slow blink |
