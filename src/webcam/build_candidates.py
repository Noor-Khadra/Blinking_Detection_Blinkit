"""Generate review proposals from MediaPipe measurements; never verified labels."""
import argparse
import csv
import json
from pathlib import Path

import numpy as np


def smooth(values, times):
    """Three-frame median within contiguous tracked regions; preserve gaps."""
    result = values.copy()
    for i in range(1, len(values) - 1):
        window = values[i - 1:i + 2]
        if np.isfinite(window).all() and np.max(np.diff(times[i - 1:i + 2])) < .15:
            result[i] = np.median(window)
    return result


def propose(times, values, ratio=.70, reference=None):
    if not len(times) or not np.isfinite(times).all() or np.any(np.diff(times) <= 0):
        raise ValueError('Timestamps must be finite and strictly increasing')
    if not 0 < ratio < 1:
        raise ValueError('Threshold ratio must be between zero and one')
    values = smooth(values, times)
    if reference is None:
        # Upper session quantile is an estimate, not a verified open-eye baseline.
        valid = values[np.isfinite(values)]
        if not len(valid):
            raise ValueError('No valid eye measurements')
        reference = float(np.percentile(valid, 80))
    if not np.isfinite(reference) or reference <= 0:
        raise ValueError('Open-eye reference must be finite and positive')
    closed = np.isfinite(values) & (values < reference * ratio)
    events = []
    i = 0
    while i < len(times):
        if not closed[i]:
            i += 1
            continue
        start = i
        while i + 1 < len(times) and closed[i + 1] and times[i + 1] - times[i] < .15:
            i += 1
        end = i
        reopened = end + 1 < len(times) and np.isfinite(values[end + 1]) and not closed[end + 1] and times[end + 1] - times[end] < .15
        offset = end + 1 if reopened else end
        peak = start + int(np.argmin(values[start:end + 1]))
        flags = []
        if end - start + 1 < 2:
            flags.append('short_candidate')
        if times[offset] - times[start] > .5:
            flags.append('long_closure_or_tracking_error')
        if start == 0 or offset == len(times) - 1:
            flags.append('recording_boundary')
        if not reopened:
            flags.append('reopening_not_observed')
        events.append(dict(event_id=len(events) + 1, onset_s=float(times[start]),
                           peak_s=float(times[peak]), offset_s=float(times[offset]),
                           onset_frame=start, peak_frame=peak, offset_frame=offset,
                           minimum_ear=float(values[peak]), event_type='closure_candidate',
                           review_status='unreviewed', flags=';'.join(flags), notes=''))
        i += 1
    return events, reference


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('measurements', type=Path)
    parser.add_argument('output_dir', type=Path)
    parser.add_argument('--eye', choices=['left', 'right', 'mean'], default='mean')
    parser.add_argument('--threshold-ratio', type=float, default=.70)
    parser.add_argument('--open-ear', type=float, help='Reference measured from verified open-eye footage')
    args = parser.parse_args()
    with args.measurements.open() as handle:
        rows = list(csv.DictReader(handle))
    times = np.array([float(r['video_time_s']) for r in rows])
    def column(name):
        return np.array([float(r[name]) if r['face_found'] == '1' and r.get(name) else np.nan for r in rows])
    left, right = column('ear_left'), column('ear_right')
    signal = {'left': left, 'right': right, 'mean': (left + right) / 2}[args.eye]
    events, reference = propose(times, signal, args.threshold_ratio, args.open_ear)
    fields = ['event_id', 'onset_s', 'peak_s', 'offset_s', 'onset_frame', 'peak_frame',
              'offset_frame', 'minimum_ear', 'event_type', 'review_status', 'flags', 'notes']
    summary = dict(frames=len(rows), candidate_events=len(events), eye=args.eye,
                   open_reference_ear=reference, threshold_ratio=args.threshold_ratio,
                   reference_method='manual' if args.open_ear is not None else 'session_80th_percentile',
                   missing_signal_frames=int((~np.isfinite(signal)).sum()),
                   time_basis='encoded_video_presentation_seconds',
                   status='UNREVIEWED; no EEG synchronization applied')
    data = dict(events=events, fields=fields, summary=summary,
                trace=[[float(t), float(l) if np.isfinite(l) else None,
                        float(r) if np.isfinite(r) else None] for t, l, r in zip(times, left, right)])
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / 'blink_candidates_UNREVIEWED.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(events)
    payload = json.dumps(data, allow_nan=False, separators=(',', ':'))
    (args.output_dir / 'review_data.json').write_text(payload)
    (args.output_dir / 'processing_summary.json').write_text(json.dumps(summary, indent=2))
    template = Path(__file__).with_name('reviewer_template.html').read_text()
    (args.output_dir / 'Review_blinks.html').write_text(template.replace('__DATA__', payload))
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
