"""Extract per-frame EAR with source presentation timestamps using PyAV."""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import time

import av
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import vision

RIGHT = [33, 160, 158, 133, 153, 144]
LEFT = [362, 385, 387, 263, 373, 380]
FIELDS = ['frame_index', 'video_time_s', 'model_timestamp_ms', 'timestamp_adjusted',
          'face_found', 'ear_right', 'ear_left', 'ear_mean', 'blink_right',
          'blink_left', 'right_eye_width_px', 'left_eye_width_px', 'eye_mid_x', 'eye_mid_y']


def ear(points, indices):
    p = points[indices]
    width = np.linalg.norm(p[0] - p[3])
    return float((np.linalg.norm(p[1] - p[5]) + np.linalg.norm(p[2] - p[4])) / (2 * width)) if width > 0 else None


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('video', type=Path)
    parser.add_argument('model', type=Path)
    parser.add_argument('output_csv', type=Path)
    args = parser.parse_args()
    for path in (args.video, args.model):
        if not path.is_file():
            parser.error(f'File does not exist: {path}')
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    options = vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=vision.RunningMode.VIDEO, num_faces=1,
        output_face_blendshapes=True)
    started = time.monotonic()
    count = found = adjusted = 0
    previous_time = None
    previous_model_ms = -1
    with av.open(str(args.video)) as container, \
            vision.FaceLandmarker.create_from_options(options) as detector, \
            args.output_csv.open('w', newline='') as handle:
        stream = container.streams.video[0]
        declared_frames = stream.frames
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for frame in container.decode(stream):
            if frame.pts is None:
                raise ValueError('Missing presentation timestamp; output is incomplete')
            timestamp = float(frame.pts * frame.time_base)
            if not np.isfinite(timestamp) or (previous_time is not None and timestamp <= previous_time):
                raise ValueError('Invalid/nonincreasing video timestamp; output is incomplete')
            previous_time = timestamp
            model_ms = max(round(timestamp * 1000), previous_model_ms + 1)
            previous_model_ms = model_ms
            changed = int(model_ms != round(timestamp * 1000))
            adjusted += changed
            rgb = frame.to_ndarray(format='rgb24')
            result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), model_ms)
            visible = bool(result.face_landmarks)
            found += visible
            row = dict(frame_index=count, video_time_s=f'{timestamp:.6f}',
                       model_timestamp_ms=model_ms, timestamp_adjusted=changed, face_found=int(visible))
            if visible:
                height, width = rgb.shape[:2]
                points = np.array([(v.x * width, v.y * height) for v in result.face_landmarks[0]])
                blend = {v.category_name: v.score for v in result.face_blendshapes[0]}
                right, left = ear(points, RIGHT), ear(points, LEFT)
                row.update(ear_right=right, ear_left=left,
                           ear_mean=(right + left) / 2 if right is not None and left is not None else None,
                           blink_right=blend.get('eyeBlinkRight'), blink_left=blend.get('eyeBlinkLeft'),
                           right_eye_width_px=float(np.linalg.norm(points[33] - points[133])),
                           left_eye_width_px=float(np.linalg.norm(points[362] - points[263])),
                           eye_mid_x=float(points[[33, 133, 362, 263], 0].mean()),
                           eye_mid_y=float(points[[33, 133, 362, 263], 1].mean()))
            writer.writerow(row)
            count += 1
            if count % 1000 == 0:
                print(f'{count} frames; {time.monotonic() - started:.1f}s', flush=True)
    if not count or (declared_frames and count != declared_frames):
        raise RuntimeError(f'Decoded {count}, declared {declared_frames}; inspect output before use')
    summary = dict(source_video=args.video.name, source_sha256=sha256(args.video),
                   model_sha256=sha256(args.model), decoded_frames=count,
                   declared_frames=declared_frames, face_found_frames=found,
                   timestamp_adjustments=adjusted, last_video_timestamp_s=previous_time,
                   time_basis='encoded_video_presentation_seconds',
                   mediapipe_version=mp.__version__, runtime_seconds=time.monotonic() - started)
    args.output_csv.with_suffix('.extraction.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == '__main__':
    main()
