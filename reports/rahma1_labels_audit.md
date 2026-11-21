# Label audit: `data/derived/noor/session4/rahma1_labels_reviewed.csv`

| Check | Count |
|---|---:|
| Labels in file | 83 |
| Accepted blinks | 80 |
| Other event types | 3 |
| Near-duplicate labels (peaks < 100 ms apart) | 11 |
| Overlapping labels | 2 |
| Blinks longer than 1.5 s | 3 |
| Blinks 0.8-1.5 s | 13 |
| Boundary/frame problems | 0 |
| Labels needing a video check | 28 |
| Labels with no issue | 0 |

Blink duration (s): median 0.383, IQR 0.267-0.585, max 71.63.

Thresholds: duplicate < 100 ms between peaks; long blink > 0.8 s; closure > 1.5 s; frame tolerance 1.5 frames at 30 fps.

## Labels to check in the video

| event_id | onset_s | duration_s | issues | suggested action |
|---|---:|---:|---|---|
| 59 | 3.633 | 0.367 | near_duplicate_of_62;placeholder_reviewer_name | keep one label, delete the duplicate |
| 62 | 3.633 | 0.367 | near_duplicate_of_59;near_duplicate_of_60;placeholder_reviewer_name | keep one label, delete the duplicate |
| 60 | 3.724 | 0.076 | near_duplicate_of_62;near_duplicate_of_61;placeholder_reviewer_name | keep one label, delete the duplicate |
| 61 | 3.733 | 0.067 | near_duplicate_of_60;placeholder_reviewer_name | keep one label, delete the duplicate |
| 74 | 99.367 | 0.233 | near_duplicate_of_3;placeholder_reviewer_name | keep one label, delete the duplicate |
| 3 | 99.433 | 0.167 | near_duplicate_of_74;placeholder_reviewer_name | keep one label, delete the duplicate |
| 80 | 185.800 | 0.333 | overlaps_81;placeholder_reviewer_name | check video: merge into one blink or fix boundaries |
| 81 | 186.067 | 0.300 | overlaps_80;placeholder_reviewer_name | check video: merge into one blink or fix boundaries |
| 5 | 210.733 | 7.267 | blink_longer_than_1.5s;placeholder_reviewer_name | check video: relabel as sustained_closure or fix offset |
| 8 | 363.200 | 0.233 | near_duplicate_of_11;placeholder_reviewer_name | keep one label, delete the duplicate |
| 11 | 363.200 | 0.100 | near_duplicate_of_8;near_duplicate_of_12;placeholder_reviewer_name | keep one label, delete the duplicate |
| 12 | 363.200 | 0.167 | near_duplicate_of_11;placeholder_reviewer_name | keep one label, delete the duplicate |
| 15 | 374.000 | 0.933 | long_blink_0.8-1.5s;near_duplicate_of_14;placeholder_reviewer_name | keep one label, delete the duplicate |
| 14 | 374.667 | 0.300 | near_duplicate_of_15;placeholder_reviewer_name | keep one label, delete the duplicate |
| 21 | 453.733 | 1.034 | long_blink_0.8-1.5s;placeholder_reviewer_name | check video: tighten offset or mark as slow blink |
| 25 | 490.900 | 1.500 | long_blink_0.8-1.5s;placeholder_reviewer_name | check video: tighten offset or mark as slow blink |
| 28 | 497.867 | 1.433 | long_blink_0.8-1.5s;placeholder_reviewer_name | check video: tighten offset or mark as slow blink |
| 29 | 501.767 | 1.400 | long_blink_0.8-1.5s;placeholder_reviewer_name | check video: tighten offset or mark as slow blink |
| 31 | 505.567 | 1.167 | long_blink_0.8-1.5s;placeholder_reviewer_name | check video: tighten offset or mark as slow blink |
| 36 | 515.100 | 1.033 | long_blink_0.8-1.5s;placeholder_reviewer_name | check video: tighten offset or mark as slow blink |
| 38 | 520.567 | 0.867 | long_blink_0.8-1.5s;placeholder_reviewer_name | check video: tighten offset or mark as slow blink |
| 41 | 528.667 | 1.033 | long_blink_0.8-1.5s;placeholder_reviewer_name | check video: tighten offset or mark as slow blink |
| 50 | 573.333 | 0.934 | long_blink_0.8-1.5s;placeholder_reviewer_name | check video: tighten offset or mark as slow blink |
| 54 | 593.000 | 1.067 | long_blink_0.8-1.5s;placeholder_reviewer_name | check video: tighten offset or mark as slow blink |
| 55 | 599.267 | 0.966 | long_blink_0.8-1.5s;placeholder_reviewer_name | check video: tighten offset or mark as slow blink |
| 56 | 603.833 | 1.700 | blink_longer_than_1.5s;placeholder_reviewer_name | check video: relabel as sustained_closure or fix offset |
| 57 | 606.033 | 1.300 | long_blink_0.8-1.5s;placeholder_reviewer_name | check video: tighten offset or mark as slow blink |
| 58 | 608.233 | 71.633 | blink_longer_than_1.5s;placeholder_reviewer_name | check video: relabel as sustained_closure or fix offset |
