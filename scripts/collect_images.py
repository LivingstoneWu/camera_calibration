#!/usr/bin/env python3
"""
Simple utility to collect still images from a list of cameras.

Each camera is sampled at ~3 Hz until the user presses the space bar,
at which point the script advances to the next camera in the list.
Captured frames are written under data/<camera_id>/ relative to the repo.
"""

import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Iterable, Union

import cv2

# Edit this list to choose which camera indexes or IDs to traverse.
CAMERA_IDS: Iterable[Union[int, str]] = [0]

CAPTURE_INTERVAL_SEC = 1.0 / 3.0  # 3 Hz
WINDOW_NAME = "Camera Preview"
SAVE_ROOT = Path(__file__).resolve().parent.parent / "data"


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

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print(f"⚠️  Failed to read a frame from camera {camera_id!r}.")
                break

            # Display the live feed so key presses can be detected.
            cv2.imshow(WINDOW_NAME, frame)

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

            key = cv2.waitKey(1) & 0xFF
            if key == ord(" "):  # Space bar moves to the next camera.
                print(f"➡️  Moving on from camera {camera_id!r}.")
                break
            if key in (27, ord("q")):  # Esc or 'q' exits the script altogether.
                print("Stopping capture (user request).")
                return True

        print(f"Finished camera {camera_id!r}: saved {saved_count} frames.")
        return False

    finally:
        cap.release()


def main() -> int:
    if not CAMERA_IDS:
        print("No camera IDs configured. Update CAMERA_IDS and re-run.")
        return 1

    SAVE_ROOT.mkdir(parents=True, exist_ok=True)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        for camera_id in CAMERA_IDS:
            want_quit = capture_from_camera(camera_id)
            if want_quit:
                break
    finally:
        cv2.destroyAllWindows()

    print("Done.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

