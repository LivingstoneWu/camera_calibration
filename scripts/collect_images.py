#!/usr/bin/env python3
"""
Simple utility to collect still images from a list of cameras.

Each camera is sampled at ~3 Hz until the user presses the space bar.
Captured frames are written under data/<camera_id>/ relative to the repo.
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Union

import cv2

CAPTURE_INTERVAL_SEC = 1.0 / 3.0  # 3 Hz
WINDOW_NAME = "Camera Preview"
SAVE_ROOT = Path(__file__).resolve().parent.parent / "data"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Collect still images from one or more cameras at ~3 Hz."
    )
    parser.add_argument(
        "camera_ids",
        nargs="+",
        help="OpenCV camera identifiers (indices or device paths).",
    )
    return parser.parse_args()


def ensure_output_directory(camera_id: Union[int, str]) -> Path:
    """Create (if needed) and return the destination directory for a camera."""
    target_dir = SAVE_ROOT / str(camera_id)
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def capture_from_camera(camera_id: Union[int, str]) -> bool:
    """Collect frames from a single camera until the space bar is pressed."""
    print(f"\nOpening camera {camera_id!r}...")
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"⚠️  Unable to open camera {camera_id!r}; skipping.")
        cap.release()
        return False

    output_dir = ensure_output_directory(camera_id)
    print(f"Saving frames under {output_dir}")
    next_capture_time = time.time()
    saved_count = 0
    recording = False

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️  Failed to read a frame from camera {camera_id!r}.")
                break

            # Display the live feed so key presses can be detected.
            cv2.imshow(WINDOW_NAME, frame)

            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):  # Esc or 'q' exits the script altogether.
                print("Stopping capture (user request).")
                return True

            if not recording:
                if key == ord(" "):  # initial space starts recording
                    recording = True
                    next_capture_time = time.time()
                    print("Recording started.")
                continue

            now = time.time()
            if now >= next_capture_time:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                filename = output_dir / f"{timestamp}.jpg"
                if cv2.imwrite(str(filename), frame):
                    saved_count += 1
                    print(f"[{camera_id}] Saved {filename.name}")
                else:
                    print(f"⚠️  Failed to write frame to {filename}")
                next_capture_time = now + CAPTURE_INTERVAL_SEC

            if key == ord(" "):  # Space bar moves to the next camera after recording
                print(f"➡️  Moving on from camera {camera_id!r}.")
                break

        print(f"Finished camera {camera_id!r}: saved {saved_count} frames.")
        return False

    finally:
        cap.release()


def main() -> int:
    args = parse_args()
    camera_ids: Iterable[Union[int, str]] = []
    for cam in args.camera_ids:
        try:
            camera_ids.append(int(cam))
        except ValueError:
            camera_ids.append(cam)

    if not camera_ids:
        print("No camera IDs provided.")
        return 1

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        for camera_id in camera_ids:
            want_quit = capture_from_camera(camera_id)
            if want_quit:
                break
    finally:
        cv2.destroyAllWindows()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

