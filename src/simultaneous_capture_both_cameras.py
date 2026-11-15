import cv2
import csv
import time
from datetime import datetime
from pathlib import Path

# Set these to your actual camera indices.
# On Windows, 0 is often the built-in webcam; 1 is often the USB camera.
WEBCAM_INDEX = 0
USB_INDEX = 1

WIDTH = 640
HEIGHT = 480  
REQUESTED_FPS = 30  # Used only for video playback timing, not as a timing assumption.

OUTPUT_DIR = Path("project_recordings")
OUTPUT_DIR.mkdir(exist_ok=True)

DATE_TAG = datetime.now().strftime("%m-%d-%Y")

WEBCAM_VIDEO = OUTPUT_DIR / f"{DATE_TAG}_webcam.avi"
USB_VIDEO = OUTPUT_DIR / f"{DATE_TAG}_usb.avi"

WEBCAM_CSV = OUTPUT_DIR / f"{DATE_TAG}_webcam_timestamps.csv"
USB_CSV = OUTPUT_DIR / f"{DATE_TAG}_usb_timestamps.csv"


def open_camera(index: int, label: str) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"{label} unavailable. Check its connection and camera index.")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, REQUESTED_FPS)

    # Read one frame to confirm the camera is actually delivering images.
    ok, frame = cap.read()
    if not ok:
        cap.release()
        raise RuntimeError(f"{label} opened but did not return a frame.")

    return cap


def make_writer(path: Path, frame) -> cv2.VideoWriter:
    height, width = frame.shape[:2]
    fourcc = cv2.VideoWriter_fourcc(*"MJPG")

    writer = cv2.VideoWriter(
        str(path),
        fourcc,
        REQUESTED_FPS,
        (width, height),
    )

    if not writer.isOpened():
        raise RuntimeError(f"Could not initialize video writer: {path}")

    return writer


def make_csv_writer(path: Path):
    csv_file = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(["frame_number", "monotonic_time_seconds", "wall_clock_iso"])
    return csv_file, writer


webcam = open_camera(WEBCAM_INDEX, "Webcam")
usb = open_camera(USB_INDEX, "USB camera")

_, webcam_frame = webcam.read()
_, usb_frame = usb.read()

webcam_writer = make_writer(WEBCAM_VIDEO, webcam_frame)
usb_writer = make_writer(USB_VIDEO, usb_frame)

webcam_csv_file, webcam_csv = make_csv_writer(WEBCAM_CSV)
usb_csv_file, usb_csv = make_csv_writer(USB_CSV)

webcam_frames = 0
usb_frames = 0

start_monotonic = time.monotonic()

try:
    while True:
        webcam_ok, webcam_frame = webcam.read()
        usb_ok, usb_frame = usb.read()

        # Timestamp each camera independently at the moment its frame arrives.
        webcam_time = time.monotonic()
        usb_time = time.monotonic()

        if webcam_ok:
            webcam_writer.write(webcam_frame)
            webcam_frames += 1
            webcam_csv.writerow([
                webcam_frames,
                f"{webcam_time:.9f}",
                datetime.now().isoformat(timespec="microseconds"),
            ])

        if usb_ok:
            usb_writer.write(usb_frame)
            usb_frames += 1
            usb_csv.writerow([
                usb_frames,
                f"{usb_time:.9f}",
                datetime.now().isoformat(timespec="microseconds"),
            ])

        if not webcam_ok and not usb_ok:
            print("Both cameras stopped delivering frames.")
            break

        # Display whichever camera(s) returned a frame.
        if webcam_ok:
            cv2.putText(
                webcam_frame,
                "Webcam recording - press Q to stop",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.imshow("Webcam", webcam_frame)

        if usb_ok:
            cv2.putText(
                usb_frame,
                "USB camera recording - press Q to stop",
                (10, 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.imshow("USB Camera", usb_frame)

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

finally:
    webcam.release()
    usb.release()
    webcam_writer.release()
    usb_writer.release()
    webcam_csv_file.close()
    usb_csv_file.close()
    cv2.destroyAllWindows()

    elapsed = time.monotonic() - start_monotonic

    print(f"Saved webcam video: {WEBCAM_VIDEO.resolve()}")
    print(f"Saved USB video:    {USB_VIDEO.resolve()}")
    print(f"Saved webcam CSV:   {WEBCAM_CSV.resolve()}")
    print(f"Saved USB CSV:      {USB_CSV.resolve()}")
    print(f"Webcam frames written: {webcam_frames}")
    print(f"USB frames written:    {usb_frames}")
    print(f"Recording duration:    {elapsed:.2f} seconds")

    if elapsed > 0:
        print(f"Webcam average rate: {webcam_frames / elapsed:.2f} frames/second")
        print(f"USB average rate:    {usb_frames / elapsed:.2f} frames/second")