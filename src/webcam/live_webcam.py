"""Preview MediaPipe face landmarks and record timestamped webcam trials."""

import argparse
import csv
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
import sys
import time

import av
import cv2
import mediapipe as mp
import numpy as np

from extract_landmarks import LEFT, RIGHT, ear

TIME_BASE = Fraction(1, 1000000)
WINDOW = "BlinkIt | Live Face Landmarks"
FIELDS = (
    "frame_index", "capture_elapsed_s", "capture_monotonic_ns", "frame_read_utc",
    "face_found", "ear_left", "ear_right", "ear_mean", "blink_left", "blink_right",
)


class Recording:
    def __init__(self, root, start_ns, width, height):
        self.start_ns = start_ns
        self.frame_index = 0
        self.last_pts = -1
        self.folder = root / datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        self.folder.mkdir(parents=True)
        self.video = av.open(str(self.folder / "webcam.mp4"), "w")
        try:
            self.stream = self.video.add_stream("libx264", rate=30)
            self.stream.width, self.stream.height = width, height
            self.stream.pix_fmt = "yuv420p"
            self.stream.codec_context.time_base = TIME_BASE
            self.stream.options = {"preset": "ultrafast", "tune": "zerolatency"}
            self.log = (self.folder / "webcam_timestamps.csv").open("w", newline="")
            self.writer = csv.DictWriter(self.log, fieldnames=FIELDS)
            self.writer.writeheader()
        except Exception:
            self.video.close()
            if hasattr(self, "log"):
                self.log.close()
            raise

    def write(self, image, capture_ns, utc, measurements):
        pts = max((capture_ns - self.start_ns) // 1000, self.last_pts + 1)
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts, frame.time_base = pts, TIME_BASE
        for packet in self.stream.encode(frame):
            self.video.mux(packet)
        self.writer.writerow({
            "frame_index": self.frame_index,
            "capture_elapsed_s": f"{pts / 1000000:.6f}",
            "capture_monotonic_ns": capture_ns,
            "frame_read_utc": utc.isoformat(timespec="microseconds"),
            **measurements,
        })
        self.last_pts = pts
        self.frame_index += 1

    def close(self):
        try:
            for packet in self.stream.encode():
                self.video.mux(packet)
        finally:
            try:
                self.video.close()
            finally:
                self.log.close()


def measurements(result, width, height):
    if not result.face_landmarks:
        return None, {"face_found": 0}

    points = np.array([(p.x * width, p.y * height) for p in result.face_landmarks[0]])
    left, right = ear(points, LEFT), ear(points, RIGHT)
    shapes = result.face_blendshapes
    scores = {item.category_name: item.score for item in shapes[0]} if shapes else {}
    return points, {
        "face_found": 1,
        "ear_left": left,
        "ear_right": right,
        "ear_mean": (left + right) / 2 if left is not None and right is not None else None,
        "blink_left": scores.get("eyeBlinkLeft"),
        "blink_right": scores.get("eyeBlinkRight"),
    }


def preview(frame, points, values, recording, fps):
    canvas = cv2.flip(frame, 1)
    height, width = canvas.shape[:2]
    if points is not None:
        px = np.rint(points).astype(np.int32)
        px[:, 0] = width - 1 - px[:, 0]
        for x, y in px:
            cv2.circle(canvas, (x, y), 1, (245, 217, 133), -1, cv2.LINE_AA)
        cv2.polylines(canvas, [px[LEFT], px[RIGHT]], True, (90, 255, 187), 2, cv2.LINE_AA)

    cv2.rectangle(canvas, (0, 0), (width, 79), (28, 24, 20), -1)
    status = "Face tracked" if points is not None else "No face detected"
    mode = f"REC  {recording.last_pts / 1000000:.1f} s" if recording else "Preview"
    color = (114, 132, 255) if recording else (199, 232, 213)
    cv2.putText(canvas, f"{mode}   |   {status}   |   {fps:.0f} FPS",
                (14, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.61, color, 2, cv2.LINE_AA)
    left, right = values.get("ear_left"), values.get("ear_right")
    detail = f"Eye openness (EAR)   L {left:.2f}   R {right:.2f}" if left is not None and right is not None else "Keep both eyes visible"
    cv2.putText(canvas, detail, (14, 57), cv2.FONT_HERSHEY_SIMPLEX, 0.53,
                (234, 238, 242), 1, cv2.LINE_AA)
    cv2.rectangle(canvas, (0, height - 35), (width, height), (28, 24, 20), -1)
    cv2.putText(canvas, "R  Start / stop recording      Q  Quit", (14, height - 11),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (234, 238, 242), 1, cv2.LINE_AA)
    return canvas


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("face_landmarker.task"))
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("data/raw/live_webcam"))
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error(f"Model missing: {args.model}. See src/webcam/README.md.")

    options = mp.tasks.vision.FaceLandmarkerOptions(
        base_options=mp.tasks.BaseOptions(model_asset_path=str(args.model)),
        running_mode=mp.tasks.vision.RunningMode.VIDEO,
        num_faces=1,
        output_face_blendshapes=True,
    )
    backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
    camera = cv2.VideoCapture(args.camera, backend)
    recording = None
    record_next = False
    window_open = False
    try:
        if not camera.isOpened():
            parser.exit(1, "Could not open camera. Check Camera access for your terminal in "
                        "System Settings > Privacy & Security, or try --camera 1.\n")
        camera.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
        start_ns = time.monotonic_ns()
        previous_ns, model_ms, fps = None, -1, 0
        with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
            cv2.namedWindow(WINDOW, cv2.WINDOW_NORMAL)
            window_open = True
            while True:
                success, frame = camera.read()
                capture_ns = time.monotonic_ns()
                utc = datetime.now(timezone.utc)
                if not success:
                    parser.exit(1, "Camera stopped returning frames.\n")
                # H.264 requires even dimensions with the yuv420p pixel format.
                height, width = frame.shape[:2]
                height, width = height - height % 2, width - width % 2
                frame = np.ascontiguousarray(frame[:height, :width])
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                model_ms = max(model_ms + 1, (capture_ns - start_ns) // 1000000)
                result = detector.detect_for_video(
                    mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), model_ms
                )
                points, values = measurements(result, width, height)
                if previous_ns is not None and capture_ns > previous_ns:
                    instant = 1e9 / (capture_ns - previous_ns)
                    fps = 0.9 * fps + 0.1 * instant if fps else instant
                previous_ns = capture_ns

                if record_next:
                    recording = Recording(args.output, capture_ns, width, height)
                    print(f"Recording to {recording.folder.resolve()}", flush=True)
                    record_next = False
                if recording:
                    recording.write(frame, capture_ns, utc, values)
                cv2.imshow(WINDOW, preview(frame, points, values, recording, fps))
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27) or cv2.getWindowProperty(WINDOW, cv2.WND_PROP_VISIBLE) == 0:
                    break
                if key in (ord("r"), ord("R")):
                    if recording:
                        finished, recording = recording, None
                        finished.close()
                        print(f"Saved {finished.frame_index} frames", flush=True)
                    else:
                        record_next = True
    finally:
        try:
            if recording:
                recording.close()
                print(f"Saved {recording.frame_index} frames", flush=True)
        finally:
            camera.release()
            if window_open:
                cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
