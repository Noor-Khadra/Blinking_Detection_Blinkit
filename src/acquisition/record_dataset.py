"""Record one BITalino channel and a laptop webcam on a shared host clock."""

import argparse
from bisect import bisect_right
from collections import deque
import csv
from datetime import datetime, timezone
from fractions import Fraction
import json
import math
from pathlib import Path
import re
import sys
from threading import Event, Thread
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
WEBCAM = ROOT / "src/webcam"
sys.path.insert(0, str(WEBCAM))
from build_candidates import propose

HZ = 1000
SECOND = 10**9
from participant_guide import ACTIVITIES, ACTIVE, BLOCK_SECONDS, BREAK, START, ParticipantView, beep, cue_at
PREVIEW_WIDTH = 440
PLOT_WIDTH = 440
PLOT_SECONDS = 10
ORDERS = {
    "A": "rest reading eye_movements voluntary talking video head_motion sustained artifacts mixed".split(),
    "B": "rest head_motion video artifacts reading sustained mixed eye_movements talking voluntary".split(),
    "C": "rest talking sustained eye_movements mixed voluntary artifacts head_motion video reading".split(),
}
ASSIGNMENTS = {"rana": "ABC", "noor": "BCA"}
DURATION = 10 * BLOCK_SECONDS
EDGES = list(range(0, DURATION + 1, BLOCK_SECONDS))


def activity_ns(elapsed_ns):
    """Accumulated active time, with pre-block pauses removed."""
    blocks, local = divmod(max(0, elapsed_ns), BLOCK_SECONDS * SECOND)
    return blocks * ACTIVE * SECOND + max(0, local - START * SECOND)


def is_activity_ns(elapsed_ns):
    return 0 <= elapsed_ns < DURATION * SECOND and elapsed_ns % (BLOCK_SECONDS * SECOND) >= START * SECOND


def recording_order(name, choice):
    if choice in ORDERS:
        return choice
    match = re.fullmatch(r"(rana|noor)_v([123])", name.lower())
    if not match:
        raise ValueError("Use a name such as rana_v1 or select order A, B or C for a custom session.")
    return ASSIGNMENTS[match[1]][int(match[2]) - 1]


BOUNDARIES = ("onset", "peak", "offset")
EAR_FIELDS = ("face_found", "ear_left", "ear_right", "ear_mean", "blink_left", "blink_right")
EVENT_FIELDS = (
    "session", "event_id", "event_type", "section", "phase", "label_source", "review_status", "flags",
    "minimum_ear", "onset_s", "peak_s", "offset_s", "onset_frame", "peak_frame", "offset_frame",
) + tuple(f"{p}_{unit}" for p in BOUNDARIES for unit in ("session_s", "sample", "eeg_s")) + ("use_for_training",)


def section(elapsed, order):
    if not 0 <= elapsed < DURATION:
        return "", ""
    i = bisect_right(EDGES, elapsed) - 1
    local = elapsed - EDGES[i]
    phase = "briefing" if local < BREAK else "countdown" if local < START else "activity"
    return ORDERS[order][i], phase


def open_csv(folder, name, fields):
    handle = (folder / name).open("w", encoding="utf-8", newline="", buffering=1)
    writer = csv.writer(handle)
    writer.writerow(fields)
    return handle, writer


def json_file(path, contents):
    pending = path.with_suffix(".tmp")
    pending.write_text(json.dumps(contents, indent=2, allow_nan=False) + "\n")
    pending.replace(path)


class Session:
    def __init__(self, root, name, port, channel, camera_index, size, ratio, order):
        import av

        if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]{0,63}", name):
            raise ValueError("Use a session name such as rana_02, with letters, numbers, underscores or hyphens.")
        self.folder = root / name
        self.folder.mkdir(parents=True)
        self.name, self.ratio, self.order = name, ratio, order
        self.webcam_count = self.eeg_count = self.eeg_gaps = 0
        self.done_camera, self.done_eeg = Event(), Event()
        self.error = None
        self.requested_stop = False
        files = []
        try:
            self.cues, self.write_cue = open_csv(self.folder, "cues.csv", ("section", "phase", "prompt", "scheduled_s", "ui_dispatch_s"))
            files.append(self.cues)
            self.frames, self.write_frame = open_csv(self.folder, "webcam_frames.csv", (
                "frame_index", "video_time_s", "session_elapsed_s", "host_monotonic_ns",
                "section", "phase", *EAR_FIELDS,
            ))
            files.append(self.frames)
            self.raw, self.write_eeg = open_csv(self.folder, "eeg.csv", (
                "sample_index", "sequence", "digital_0", "digital_1", "digital_2", "digital_3", "adc",
            ))
            files.append(self.raw)
            self.clock, self.write_clock = open_csv(self.folder, "eeg_clock.csv", (
                "sample_index", "batch_received_session_s", "read_duration_ms",
            ))
            files.append(self.clock)
            self.video = av.open(str(self.folder / "webcam.mp4"), "w", options={
                "movflags": "+frag_keyframe+empty_moov+default_base_moof", "frag_duration": "1000000",
            })
            files.append(self.video)
            self.stream = self.video.add_stream("libx264", rate=30)
            self.stream.width, self.stream.height = size
            self.stream.pix_fmt = "yuv420p"
            self.time_base = Fraction(1, 1000000)
            self.stream.codec_context.time_base = self.time_base
            self.stream.options = {"preset": "ultrafast", "tune": "zerolatency"}
            self.first_camera_ns = None
            self.last_pts = -1
            self.start_ns = time.monotonic_ns()
            self.end_ns = self.start_ns + DURATION * SECOND
            self.meta = {
                "session": name, "state": "recording", "planned_duration_s": 10 * ACTIVE, "planned_wall_duration_s": DURATION,
                "active_only": True, "protocol_order": order,
                "session_start_utc": datetime.now(timezone.utc).isoformat(),
                "session_start_monotonic_ns": self.start_ns, "eeg_sampling_rate_hz": HZ,
                "eeg_address": port, "eeg_channel": f"A{channel}", "camera_index": camera_index,
                "eeg_clock_basis": "device sample sequence fitted to host batch receipt times",
                "webcam_clock_basis": "host frame receipt time",
                "hardware_sync_verified": False, "ear_threshold_ratio": ratio,
            }
            json_file(self.folder / "session.json", self.meta)
            with (self.folder / "protocol.csv").open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(("section", "activity", "instruction", "briefing_start_s", "countdown_start_s", "activity_start_s", "end_s"))
                for i, key in enumerate(ORDERS[order]):
                    title, instruction = ACTIVITIES[key]
                    writer.writerow((key, title, instruction, EDGES[i], EDGES[i] + BREAK, EDGES[i] + START, EDGES[i + 1]))
        except Exception:
            for handle in reversed(files):
                handle.close()
            raise

    def stop(self, error=None):
        self.requested_stop = True
        self.error = error or self.error
        self.end_ns = min(self.end_ns, time.monotonic_ns())

    def camera_frame(self, image, host_ns, values):
        import av

        elapsed_ns = host_ns - self.start_ns
        if not is_activity_ns(elapsed_ns):
            return
        active_ns = activity_ns(elapsed_ns)
        if self.first_camera_ns is None:
            self.first_camera_ns = active_ns
        pts = max((active_ns - self.first_camera_ns) // 1000, self.last_pts + 1)
        frame = av.VideoFrame.from_ndarray(image, format="bgr24")
        frame.pts, frame.time_base = pts, self.time_base
        for packet in self.stream.encode(frame):
            self.video.mux(packet)
        elapsed = (host_ns - self.start_ns) / SECOND
        self.write_frame.writerow((
            self.webcam_count, f"{pts / 1000000:.6f}", f"{elapsed:.9f}", host_ns,
            *section(elapsed, self.order), *(values.get(key, "") for key in EAR_FIELDS),
        ))
        self.webcam_count += 1
        self.last_pts = pts

    def close_camera(self):
        try:
            if self.webcam_count:
                for packet in self.stream.encode():
                    self.video.mux(packet)
        finally:
            try:
                self.video.close()
            finally:
                self.frames.close()
                self.done_camera.set()

    def close_eeg(self):
        try:
            self.raw.close()
            self.clock.close()
        finally:
            self.done_eeg.set()

    def finalize(self):
        self.cues.close()
        if not self.webcam_count or not self.eeg_count:
            self.error = self.error or "At least one stream did not save any samples."
        complete = not self.error and not self.requested_stop
        try:
            summary = export_labels(self.folder, completed=complete)
        except Exception as exc:
            self.error = self.error or f"Could not export labels: {exc}"
            summary = {}
        duration = max(0, (self.end_ns - self.start_ns) / SECOND)
        self.meta.update(summary, state="complete" if not self.error and not self.requested_stop else "partial",
                         wall_duration_s=duration, recorded_duration_s=activity_ns(self.end_ns - self.start_ns) / SECOND, webcam_frames=self.webcam_count,
                         eeg_rows=self.eeg_count, estimated_missing_eeg_samples=self.eeg_gaps,
                         error=self.error)
        json_file(self.folder / "session.json", self.meta)


def read_rows(path):
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def clock_map(anchors):
    """Fit rate from per-five-second low-delay observations, then estimate intercept."""
    y = np.array([float(row["batch_received_session_s"]) for row in anchors])
    x = np.array([int(row["sample_index"]) for row in anchors]) / HZ
    if not len(x):
        return None
    offsets = y - x
    knot = [int(np.argmin(offsets[i:i+100])) + i for i in range(0, len(x), 100)]
    drift = float(np.polyfit(x[knot], offsets[knot], 1)[0]) if len(knot) >= 3 else 0.
    plausible = abs(drift) <= .005
    slope = (1 + drift) if plausible else 1.
    intercept = float(np.quantile(y - x * slope, .1))
    jitter = y - (x * slope + intercept)
    return slope, intercept, float(np.quantile(jitter, .95)), plausible


def auto_coverage(folder, name, frames, times, eyes, events, eeg, eeg_times, threshold):
    rows = []
    if threshold is not None:
        sections = np.array([row["section"] if row["phase"] == "activity" and row["section"] != "eyes_closed"
                             else "" for row in frames])
        open_eyes = (sections != "") & np.all(np.isfinite(eyes[:, :2]) & (eyes[:, :2] >= threshold), axis=1)
        for event in events:
            start, end = np.searchsorted(times, [float(event["onset_session_s"]) - .15,
                                                float(event["offset_session_s"]) + .15])
            open_eyes[start:end] = False
        split = np.r_[0, np.flatnonzero((np.diff(times) >= .15) | (sections[:-1] != sections[1:])
                                        | (open_eyes[:-1] != open_eyes[1:])) + 1, len(times)]
        for a, b in zip(split[:-1], split[1:]):
            if not open_eyes[a] or times[b-1] - times[a] < 1:
                continue
            first, after = np.searchsorted(eeg_times, [times[a], times[b-1]])
            if after - first >= HZ:
                rows.append((name, sections[a], "mediapipe_ear_auto", int(eeg[first]), int(eeg[after-1]) + 1))
    with (folder / "no_blink_coverage_auto.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(("session", "section", "label_source", "start_sample", "end_sample"))
        writer.writerows(rows)
    return len(rows)


def export_labels(folder, reviewed=None, alignment_reviewed=False, *, completed=None):
    frames = read_rows(folder / "webcam_frames.csv")
    anchors = read_rows(folder / "eeg_clock.csv")
    model = clock_map(anchors)
    samples = (np.loadtxt(folder / "eeg.csv", delimiter=",", skiprows=1, usecols=(0, 6), dtype=np.int64, ndmin=2)
               if anchors else np.empty((0, 2), dtype=np.int64))
    eeg = samples[:, 0]
    if len(eeg) and model:
        slope, intercept, residual, plausible = model
        eeg_times = eeg / HZ * slope + intercept
    else:
        eeg_times = np.array([])
    if len(eeg_times) and not reviewed:
        np.savez_compressed(folder / "eeg_timeline.npz", adc=samples[:, 1].astype(np.int16),
                            sample_index=eeg, session_elapsed_s=eeg_times, nominal_rate_hz=HZ)
    video_times = np.array([float(row["video_time_s"]) for row in frames])
    session_times = np.array([float(row["session_elapsed_s"]) for row in frames])
    eyes = np.array([[float(row[key]) if row[key] else np.nan for key in ("ear_left", "ear_right", "ear_mean")] for row in frames])
    reference = None
    meta = json.loads((folder / "session.json").read_text())
    complete = meta["state"] == "complete" if completed is None else completed
    usable_timing = bool(len(eeg_times) and model[3] and model[2] <= .05)
    valid = bool(len(frames) and np.isfinite(eyes[:, 2]).any())
    if reviewed:
        events = [e for e in read_rows(reviewed) if e["review_status"] == "accepted"
                  and e.get("human_review_status") == "completed"
                  and e["event_type"] in ("blink", "sustained_closure")]
    elif valid:
        ratio = meta["ear_threshold_ratio"]
        if meta.get("active_only"):
            reference = float(np.nanpercentile(eyes[:, 2], 80))
            events = []
            boundaries = [i for i in range(1, len(frames)) if frames[i]["section"] != frames[i-1]["section"]]
            for first, after in zip([0] + boundaries, boundaries + [len(frames)]):
                candidates, _ = propose(video_times[first:after], eyes[first:after, 2], ratio, reference)
                for candidate in candidates:
                    for point in BOUNDARIES:
                        candidate[f"{point}_frame"] += first
                events.extend(candidates)
        else:
            events, reference = propose(video_times, eyes[:, 2], ratio)
        for e in events:
            e["event_type"] = "sustained_closure" if e["offset_s"] - e["onset_s"] > .5 else "blink"
            if e["offset_s"] - e["onset_s"] > .5:
                e["flags"] = ";".join(filter(None, (e["flags"], "verify_sustained_closure")))
    else:
        events = []
    output = []
    for event in sorted(events, key=lambda e: float(e["peak_s"])):
        e = dict(event)
        e["session"] = meta["session"]
        e["event_id"] = len(output) + 1
        e["label_source"] = "human_reviewed" if reviewed else "mediapipe_ear_auto"
        good = usable_timing
        if any(float(e[f"{p}_s"]) > float(e[f"{q}_s"]) for p, q in zip(BOUNDARIES, BOUNDARIES[1:])):
            raise ValueError("Review boundaries must be in onset, peak, offset order.")
        for point in BOUNDARIES:
            i = int(e[f"{point}_frame"])
            if not 0 <= i < len(frames):
                raise ValueError(f"{point} frame {i} is outside the recording")
            t = session_times[i]
            if not math.isclose(float(e[f"{point}_s"]), video_times[i], abs_tol=.08, rel_tol=0):
                raise ValueError(f"{point} timestamp does not match frame {i}; verify the exported CSV")
            e[f"{point}_session_s"] = f"{t:.9f}"
            j = min(int(np.searchsorted(eeg_times, t)), len(eeg_times) - 1)
            if j > 0 and abs(eeg_times[j-1] - t) <= abs(eeg_times[j] - t):
                j -= 1
            if j < 0 or not eeg_times[0] <= t <= eeg_times[-1] or abs(eeg_times[j] - t) > .003:
                good = False
                continue
            e[f"{point}_sample"] = int(eeg[j])
            e[f"{point}_eeg_s"] = f"{eeg_times[j]:.9f}"
        peak = int(e["peak_frame"])
        e["section"], e["phase"] = (frames[peak]["section"], frames[peak]["phase"])
        crosses = len({(frames[int(e[f"{p}_frame"])]["section"], frames[int(e[f"{p}_frame"])]["phase"]) for p in BOUNDARIES}) > 1
        if crosses:
            e["flags"] = ";".join(filter(None, (e.get("flags"), "crosses_protocol_boundary")))
        e["use_for_training"] = int(bool(complete and good and not crosses and e["phase"] == "activity"
                                         and (not reviewed or alignment_reviewed)))
        output.append(e)
    name = "eeg_events_reviewed.csv" if reviewed else "eeg_events_auto.csv"
    with (folder / name).open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, EVENT_FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(output)
    coverage_count = None
    if not reviewed:
        threshold = reference * meta["ear_threshold_ratio"] if reference is not None and usable_timing and complete else None
        coverage_count = auto_coverage(folder, meta["session"], frames, session_times, eyes, output,
                                       eeg, eeg_times, threshold)
    if not reviewed and len(frames):
        trace = [[float(t), *[float(v) if np.isfinite(v) else None for v in eye[:2]]] for t, eye in zip(video_times, eyes)]
        fields = ["event_id", "onset_s", "peak_s", "offset_s", "onset_frame", "peak_frame", "offset_frame",
                  "minimum_ear", "event_type", "review_status", "flags", "notes", "reviewer_name",
                  "human_review_status", "boundary_status", "reviewer_kind"]
        payload = json.dumps({"events": output, "fields": fields, "trace": trace}, allow_nan=False, separators=(",", ":"))
        template = (WEBCAM / "reviewer_template.html").read_text(encoding="utf-8")
        template = template.replace("video.mp4", "webcam.mp4").replace("Accept blink", "Accept selected class")
        template = template.replace("e.event_type='blink';e.review_status", "if(!['blink','sustained_closure'].includes(e.event_type))return alert('Select blink or sustained closure.');e.review_status")
        (folder / "Review_blinks.html").write_text(template.replace("__DATA__", payload), encoding="utf-8")
    return {"automatic_candidates": len(output), "open_eye_reference_ear": reference,
            "automatic_events_eligible": sum(e["use_for_training"] for e in output),
            "automatic_no_blink_intervals": coverage_count,
            "frames_with_ear": int(np.isfinite(eyes[:, 2]).sum()) if len(frames) else 0,
            "webcam_span_session_s": [float(session_times[0]), float(session_times[-1])] if len(frames) else None,
            "eeg_span_session_s": [float(eeg_times[0]), float(eeg_times[-1])] if len(eeg_times) else None,
            "clock_map": dict(rate_scale=model[0], intercept_s=model[1], batch_residual_p95_s=model[2],
                              fitted_rate_plausible=model[3]) if model else None}


class Devices:
    def __init__(self, port, channel, camera_index, model):
        self.port, self.channel, self.camera_index, self.model = port, channel, camera_index, model
        self.session = None
        self.quit = Event()
        self.camera_ns = self.eeg_ns = 0
        self.preview = None
        self.eyes = None
        self.eeg_history = deque(maxlen=240)
        self.size = None
        self.camera_error = self.eeg_error = None
        self.threads = [Thread(target=self.eeg_loop, daemon=True)]
        if camera_index is not None:
            self.threads.insert(0, Thread(target=self.camera_loop, daemon=True))
        for worker in self.threads:
            worker.start()

    def ready(self):
        now = time.monotonic_ns()
        return self.size and now - self.camera_ns < 2 * SECOND and now - self.eeg_ns < 2 * SECOND and not self.error

    @property
    def error(self):
        return self.camera_error or self.eeg_error

    def camera_loop(self):
        import cv2
        import mediapipe as mp
        from live_webcam import measurements, preview

        cap = None
        try:
            backend = cv2.CAP_AVFOUNDATION if sys.platform == "darwin" else cv2.CAP_ANY
            cap = cv2.VideoCapture(self.camera_index, backend)
            if not cap.isOpened():
                raise RuntimeError("Allow camera access for Terminal or select another camera index.")
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)
            options = mp.tasks.vision.FaceLandmarkerOptions(
                base_options=mp.tasks.BaseOptions(model_asset_path=str(self.model)),
                running_mode=mp.tasks.vision.RunningMode.VIDEO, num_faces=1, output_face_blendshapes=True,
            )
            previous_ms = -1
            clock_start = time.monotonic_ns()
            fps = 0.
            with mp.tasks.vision.FaceLandmarker.create_from_options(options) as detector:
                while not self.quit.is_set():
                    ok, image = cap.read()
                    now = time.monotonic_ns()
                    if not ok:
                        raise RuntimeError("Camera stopped returning frames.")
                    if self.size is None:
                        w, h = image.shape[1] // 2 * 2, image.shape[0] // 2 * 2
                        self.size = (w, h)
                    w, h = self.size
                    image = np.ascontiguousarray(image[:h, :w])
                    if image.shape[:2] != (h, w):
                        raise RuntimeError("Camera resolution changed during the trial.")
                    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    model_ms = max((now - clock_start) // 1000000, previous_ms + 1)
                    previous_ms = model_ms
                    result = detector.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), model_ms)
                    points, values = measurements(result, w, h)
                    trial = self.session
                    if trial and not trial.done_camera.is_set():
                        if now >= trial.end_ns:
                            trial.close_camera()
                        elif now >= trial.start_ns:
                            trial.camera_frame(image, now, values)
                    if self.camera_ns and now > self.camera_ns:
                        instant = SECOND / (now - self.camera_ns)
                        fps = fps * .9 + instant * .1 if fps else instant
                    canvas = preview(image, points, values, None, fps)
                    scaled = cv2.resize(canvas[:h-35], (PREVIEW_WIDTH, max(1, round((h-35)*PREVIEW_WIDTH/w))))
                    self.preview = cv2.cvtColor(scaled, cv2.COLOR_BGR2RGB)
                    self.eyes = (now, values.get("ear_left"), values.get("ear_right"))
                    self.camera_ns = now
        except Exception as exc:
            self.camera_error = f"Camera: {exc}"
        finally:
            trial = self.session
            if trial and not trial.done_camera.is_set():
                if self.camera_error:
                    trial.stop(self.camera_error)
                trial.close_camera()
            if cap:
                cap.release()

    def eeg_loop(self):
        board = None
        started = False
        stage = "opening the port and reading board firmware"
        try:
            from bitalino import BITalino
            board = BITalino(self.port, timeout=5)
            board.timeout = 2
            stage = f"starting A{self.channel} acquisition"
            board.start(HZ, [self.channel - 1])
            started = True
            stage = "waiting for the first EEG samples"
            previous = None
            index = -1
            offsets = deque(maxlen=40)
            for _ in range(10):
                if self.quit.is_set():
                    return
                rows = board.read(50)
                end = time.monotonic_ns()
                for row in rows:
                    seq = int(row[0])
                    step = ((seq - previous) % 16 or 16) if previous is not None else 1
                    index += step
                    previous = seq
                offsets.append(end - index * (SECOND // HZ))
            host_origin = min(offsets)

            stage = "reading the EEG stream"
            while not self.quit.is_set():
                read_start = time.monotonic_ns()
                rows = board.read(50)
                end = time.monotonic_ns()
                self.eeg_ns = end
                trial = self.session
                saving = trial and not trial.done_eeg.is_set()
                saved_batch = False
                sample_indices = np.empty(len(rows), dtype=np.int64)
                for row_number, row in enumerate(rows):
                    seq = int(row[0])
                    step = ((seq - previous) % 16 or 16) if previous is not None else 1
                    index += step
                    previous = seq
                    sample_indices[row_number] = index
                    if not saving:
                        continue
                    sample_host = host_origin + index * (SECOND // HZ)
                    if sample_host < trial.end_ns and is_activity_ns(sample_host - trial.start_ns):
                        trial.write_eeg.writerow((index, *(int(value) for value in row)))
                        trial.eeg_count += 1
                        trial.eeg_gaps += step - 1
                        saved_batch = True
                offsets.append(end - index * (SECOND // HZ))
                host_origin = min(offsets)
                self.eeg_history.append((host_origin + sample_indices * (SECOND // HZ), rows[:, 5].copy()))
                if saved_batch:
                    trial.write_clock.writerow((
                        index, f"{(end-trial.start_ns)/SECOND:.9f}", f"{(end-read_start)/1000000:.3f}",
                    ))
                if saving and end >= trial.end_ns and host_origin + index*(SECOND//HZ) >= trial.end_ns:
                    trial.close_eeg()
                elif saving and end >= trial.end_ns + SECOND:
                    trial.stop("EEG sample clock did not reach the final boundary.")
                    trial.close_eeg()
        except Exception as exc:
            self.eeg_error = f"BITalino ({stage}): {exc}"
        finally:
            trial = self.session
            if trial and not trial.done_eeg.is_set():
                if self.eeg_error:
                    trial.stop(self.eeg_error)
                trial.close_eeg()
            if board is not None:
                try:
                    if started:
                        board.stop()
                except Exception:
                    pass
                finally:
                    board.close()


def eeg_plot_columns(history, end_ns, column_count):
    """Return each display pixel's raw ADC extrema, preserving brief excursions."""
    if not history:
        return ()
    span = PLOT_SECONDS * SECOND
    times = np.concatenate([times for times, _ in history])
    values = np.concatenate([values for _, values in history])
    visible = (end_ns - span <= times) & (times <= end_ns)
    columns = ((times[visible] - (end_ns - span)) * (column_count - 1) // span).astype(int)
    lo = np.full(column_count, np.inf)
    hi = np.full(column_count, -np.inf)
    np.minimum.at(lo, columns, values[visible])
    np.maximum.at(hi, columns, values[visible])
    return [(int(i), int(lo[i]), int(hi[i])) for i in np.flatnonzero(np.isfinite(lo))]


def serial_port_choices():
    if sys.platform == "win32":
        from serial.tools.list_ports import comports

        return [port.device for port in sorted(comports())]
    paths = (p for p in Path("/dev").glob("cu.*")
             if p.name not in {"cu.Bluetooth-Incoming-Port", "cu.debug-console"})
    return [str(p) for p in sorted(paths, key=lambda p: (not p.name.lower().startswith("cu.bitalino"), p.name))]


def test_eeg(port, channel, output):
    print(f"Testing {port}, A{channel}. The camera stays off.", flush=True)
    devices = Devices(port, channel, None, None)
    report = {"port": port, "channel": f"A{channel}", "utc": datetime.now(timezone.utc).isoformat()}
    try:
        deadline = time.monotonic() + 15
        while len(devices.eeg_history) < 40 and not devices.eeg_error and time.monotonic() < deadline:
            time.sleep(.1)
        batches = devices.eeg_history.copy()
        report["samples_received"] = sum(len(values) for _, values in batches)
        report["error"] = devices.eeg_error or ("Did not receive two seconds of EEG samples." if len(batches) < 40 else None)
    finally:
        devices.quit.set()
        devices.threads[0].join(timeout=3)
    output.mkdir(parents=True, exist_ok=True)
    log = output / "eeg_connection_test.json"
    json_file(log, report)
    print(report["error"] or f"EEG streaming verified: {report['samples_received']} samples.", flush=True)
    print(f"Saved diagnostics to {log}", flush=True)
    return 1 if report["error"] else 0


def run_app(args):
    import tkinter as tk
    from tkinter import ttk
    from PIL import Image, ImageTk

    root = tk.Tk()
    root.title("BlinkIt · Dataset recorder")
    root.minsize(950, 440)
    shell = ttk.Frame(root, padding=18)
    shell.grid(sticky="nsew")
    root.columnconfigure(0, weight=1)
    shell.columnconfigure(1, weight=1)
    name = tk.StringVar(value=args.session_name)
    order_choice = tk.StringVar(value="Auto from name")
    port = tk.StringVar(value=args.port or "")
    channel = tk.StringVar(value=str(args.channel))
    port_field = ttk.Combobox(shell, textvariable=port)

    def refresh_ports():
        ports = serial_port_choices()
        port_field.config(values=ports)
        bit_ports = [path for path in ports if Path(path).name.lower().startswith("cu.bitalino")]
        if not port.get().strip() and len(bit_ports) == 1:
            port.set(bit_ports[0])

    ttk.Button(shell, text="Test countdown sound", command=lambda: beep(root)).grid(row=8, column=0, sticky="w")
    port_field.config(postcommand=refresh_ports)
    refresh_ports()
    fields = [
        ("Session name", ttk.Entry(shell, textvariable=name)),
        ("BITalino port", port_field),
        ("EEG channel", ttk.Combobox(shell, textvariable=channel, values=["1", "2", "3", "4"], state="readonly")),
    ]
    for i, (label, field) in enumerate(fields):
        ttk.Label(shell, text=label).grid(row=i, column=0, sticky="w", padx=(0, 10), pady=3)
        field.grid(row=i, column=1, columnspan=2, sticky="ew", pady=3)
    fields[0][1].grid_configure(columnspan=1)
    ttk.Combobox(shell, textvariable=order_choice, values=("Auto from name", "A", "B", "C"),
                 width=15, state="readonly").grid(row=0, column=2)
    status = ttk.Label(shell, text="Connect the laptop camera and BITalino to begin.", wraplength=560)
    status.grid(row=4, column=0, columnspan=3, sticky="w", pady=(9, 4))
    task = ttk.Label(shell, font=("TkDefaultFont", 20), wraplength=560)
    task.grid(row=5, column=0, columnspan=3, sticky="w")
    instructions = ttk.Label(shell, wraplength=560)
    instructions.grid(row=6, column=0, columnspan=3, sticky="w")
    visuals = ttk.Frame(shell)
    visuals.grid(row=7, column=0, columnspan=3, sticky="nw", pady=6)
    view = ttk.Label(visuals)
    view.grid(row=0, column=0, rowspan=2, sticky="n", padx=(0, 12))
    eeg_chart = tk.Canvas(visuals, width=PLOT_WIDTH, height=158, background="#132015", highlightthickness=0)
    eeg_chart.grid(row=0, column=1, sticky="nw", pady=(0, 6))
    chart = tk.Canvas(visuals, width=PLOT_WIDTH, height=158, background="#132015", highlightthickness=0)
    chart.grid(row=1, column=1, sticky="nw")
    ear_history = deque()
    last_preview = None
    last_plot_ns = 0
    devices = trial = finishing = participant = None
    closing = False
    shown = None
    configured = False

    def connect():
        nonlocal devices, configured
        refresh_ports()
        if not port.get().strip():
            status.config(text="Select or enter the paired BITalino's serial port.")
            return
        if devices:
            devices.quit.set()
            if any(worker.is_alive() for worker in devices.threads):
                status.config(text="Previous connection is closing. Try again in a moment.")
                return
        configured = True
        ear_history.clear()
        devices = Devices(port.get().strip(), int(channel.get()), args.camera, args.model)
        status.config(text="Connecting devices…")

    def start():
        nonlocal trial, shown, participant
        if not devices or not devices.ready():
            return
        try:
            order = recording_order(name.get().strip(), order_choice.get())
            participant = ParticipantView(root, stop)
            trial = Session(args.output, name.get().strip(), devices.port, devices.channel,
                            devices.camera_index, devices.size, args.ear_ratio,
                            order)
        except FileExistsError:
            participant.close()
            participant = None
            status.config(text="That session name already exists. Choose a new name.")
            return
        except (OSError, RuntimeError, ValueError) as exc:
            if participant:
                participant.close()
                participant = None
            status.config(text=str(exc))
            return
        shown = None
        devices.session = trial

    def stop():
        if trial:
            trial.stop()

    def close():
        nonlocal closing
        closing = True
        if trial:
            trial.stop()
        elif not finishing:
            if devices:
                devices.quit.set()
            root.destroy()

    connect_button = ttk.Button(shell, text="Connect devices", command=connect)
    start_button = ttk.Button(shell, text="Start recording", command=start)
    stop_button = ttk.Button(shell, text="Stop early", command=stop)
    for i, button in enumerate((connect_button, start_button, stop_button)):
        button.grid(row=3, column=i, sticky="w", pady=(8, 0))

    left_px, right_px = 42, PLOT_WIDTH - 12
    top_px, bottom_px = 29, 133

    def axes(target, title, lo, hi):
        target.delete("all")
        target.create_text(12, 13, text=title, fill="#d2e5d7", anchor="w")
        for i in range(3):
            y = top_px + (bottom_px - top_px) * i / 2
            target.create_line(left_px, y, right_px, y, fill="#2d4a35")
            value = hi - (hi-lo) * i / 2
            target.create_text(left_px-5, y, text=f"{value:g}", fill="#adc4b3", anchor="e")
        for x, text in ((left_px, "−10 s"), ((left_px + right_px)/2, "−5 s"), (right_px, "Now")):
            target.create_text(x, 146, text=text, fill="#adc4b3")

    def draw_ear(now):
        cutoff = now - PLOT_SECONDS * SECOND
        while ear_history and ear_history[0][0] < cutoff:
            ear_history.popleft()
        top = max(.6, max((v for row in ear_history for v in row[1:] if v is not None), default=0.))
        top = math.ceil((top - 1e-9) * 10) / 10
        axes(chart, "EAR", 0, top)
        chart.create_text(55, 13, text="Left", fill="#8eefff", anchor="w")
        chart.create_text(89, 13, text="Right", fill="#a3ffb9", anchor="w")
        if not ear_history:
            chart.create_text(PLOT_WIDTH/2, 80, text="Waiting for webcam landmarks", fill="#adc4b3")
            return
        def x_at(t):
            return left_px + (t - cutoff) / (PLOT_SECONDS * SECOND) * (right_px - left_px)
        for i, color in ((1, "#8eefff"), (2, "#a3ffb9")):
            last_t = None
            last_y = None
            for row in ear_history:
                if row[i] is None:
                    last_t = None
                    continue
                x = x_at(row[0])
                y = bottom_px - row[i] / top * (bottom_px - top_px)
                if last_t is not None and row[0] - last_t < .25 * SECOND:
                    chart.create_line(x_at(last_t), last_y, x, y, fill=color)
                last_t, last_y = row[0], y

    def draw_eeg(now):
        columns = eeg_plot_columns(devices.eeg_history.copy(), now, right_px-left_px+1) if devices else ()
        lo, hi = 0, 1023
        if columns:
            low = min(v[1] for v in columns)
            high = max(v[2] for v in columns)
            center = (low+high)//2
            lo, hi = max(0, min(low, center-16)), min(1023, max(high, center+16))
        title = f"EEG A{devices.channel} · Raw ADC" if devices else "EEG · Raw ADC"
        axes(eeg_chart, title, lo, hi)
        if not columns:
            eeg_chart.create_text(PLOT_WIDTH/2, 80, text="Waiting for EEG samples", fill="#adc4b3")
            return
        def y_at(v):
            return bottom_px - (v-lo)/(hi-lo)*(bottom_px-top_px)
        for i, low, high in columns:
            x = left_px+i
            y = y_at(high)
            eeg_chart.create_line(x, y, x, max(y+1, y_at(low)), fill="#f4c27e")

    def refresh():
        nonlocal trial, finishing, shown, configured, last_preview, last_plot_ns, participant
        latest_preview = devices.preview if devices else None
        if latest_preview is not None and latest_preview is not last_preview:
            last_preview = latest_preview
            view.image = ImageTk.PhotoImage(Image.fromarray(latest_preview))
            view.config(image=view.image)
            if devices.eyes and (not ear_history or devices.eyes[0] > ear_history[-1][0]):
                ear_history.append(devices.eyes)
        now = time.monotonic_ns()
        if now - last_plot_ns >= 120000000:
            draw_ear(now)
            draw_eeg(now)
            last_plot_ns = now
        if trial:
            elapsed = (time.monotonic_ns() - trial.start_ns) / SECOND
            if devices.error:
                trial.stop(devices.error)
                if not devices.threads[0].is_alive() and not trial.done_camera.is_set():
                    trial.close_camera()
                if not devices.threads[1].is_alive() and not trial.done_eeg.is_set():
                    trial.close_eeg()
            if trial.done_camera.is_set() and trial.done_eeg.is_set():
                participant.close()
                participant = None
                finishing = Thread(target=trial.finalize, daemon=True)
                finishing.trial = trial
                finishing.start()
                trial = None
                status.config(text="Saving and generating blink labels…")
            elif elapsed < DURATION and not trial.requested_stop:
                i = bisect_right(EDGES, elapsed) - 1
                key = ORDERS[trial.order][i]
                title, action = ACTIVITIES[key]
                local = elapsed - EDGES[i]
                phase, scheduled, prompt = cue_at(key, local)
                status.config(text=f"{trial.name} · Order {trial.order} · Block {i+1}/10 · {int(elapsed)//60:02d}:{int(elapsed)%60:02d}")
                task.config(text=title)
                instructions.config(text=prompt)
                participant.render(key, i+1, local, phase, prompt)
                if shown != (i, scheduled):
                    if phase == "countdown" or scheduled == START or prompt == "Open your eyes":
                        beep(root, high=phase == "activity")
                    trial.write_cue.writerow((key, phase, prompt, EDGES[i] + scheduled, f"{elapsed:.6f}"))
                    shown = (i, scheduled)
            else:
                status.config(text="Stopping both streams…")
        elif finishing and not finishing.is_alive():
            result = finishing.trial
            finishing = None
            status.config(text=f"{result.meta['state'].capitalize()} · {result.folder}. {result.error or ''}")
            task.config(text="Recording saved")
            instructions.config(text="Automatic labels and no-blink coverage are in the session folder.")
            if closing:
                close()
                return
        elif not finishing and devices and devices.error:
            if configured:
                print(devices.error, flush=True)
            status.config(text=devices.error)
            configured = False
        elif configured and not finishing and devices.ready():
            print("Camera and EEG streams ready.", flush=True)
            status.config(text="Both streams ready. Enter a session name and start recording.")
            configured = False
        ready = bool(devices and devices.ready())
        idle = not (trial or finishing)
        connect_button.config(state="normal" if idle else "disabled")
        start_button.config(state="normal" if idle and ready else "disabled")
        stop_button.config(state="normal" if trial and not trial.requested_stop else "disabled")
        root.after(60, refresh)

    root.protocol("WM_DELETE_WINDOW", close)
    refresh()
    if args.connect:
        root.after_idle(connect)
    root.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--session-name", default="", help="Prefill the recording session name")
    parser.add_argument("--connect", action="store_true", help="Connect devices when the window opens")
    parser.add_argument("--port", help="BITalino serial device, for example /dev/cu.BITalino")
    parser.add_argument("--test-eeg", action="store_true", help="Run an EEG connection check with the camera off")
    parser.add_argument("--channel", type=int, choices=range(1, 5), default=4)
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--model", type=Path, default=ROOT / "face_landmarker.task")
    parser.add_argument("--output", type=Path, default=ROOT / "data/raw/datasets")
    parser.add_argument("--ear-ratio", type=float, default=.55)
    parser.add_argument("--reviewed-labels", nargs=2, type=Path, metavar=("SESSION_DIR", "CSV"))
    parser.add_argument("--auto-labels", type=Path, metavar="SESSION_DIR",
                        help="Rebuild automatic labels and coverage for an existing session")
    parser.add_argument("--alignment-reviewed", action="store_true",
                        help="Mark reviewed activity events as eligible after checking stream alignment")
    args = parser.parse_args()
    if args.test_eeg:
        ports = [name for name in serial_port_choices() if Path(name).name.lower().startswith("cu.bitalino")]
        if not args.port and len(ports) != 1:
            parser.error("Select the BITalino for the test with --port.")
        raise SystemExit(test_eeg(args.port or ports[0], args.channel, args.output))
    elif args.auto_labels:
        folder = args.auto_labels
        summary = export_labels(folder)
        path = folder / "session.json"
        json_file(path, {**json.loads(path.read_text()), **summary})
        print(folder / "eeg_events_auto.csv")
    elif args.reviewed_labels:
        folder, labels = args.reviewed_labels
        export_labels(folder, labels, args.alignment_reviewed)
        print(folder / "eeg_events_reviewed.csv")
    else:
        if not args.model.is_file():
            parser.error(f"Missing face landmarker model: {args.model}")
        if not 0 < args.ear_ratio < 1:
            parser.error("--ear-ratio must be between 0 and 1")
        run_app(args)


if __name__ == "__main__":
    main()
