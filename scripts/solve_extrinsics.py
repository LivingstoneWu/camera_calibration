#!/usr/bin/env python3
"""
Interactive extrinsics solver that reuses the Zhang 2000 helpers.

The script loads intrinsics from `intrinsics/<camera_id>/calibration.txt`, streams frames
from the requested camera, and upon user request computes the pose of the checkerboard in
front of the camera using `Zhang2000Calib.get_rvec_and_tvec`. Results are written in a
human-readable text format.
"""

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

from algorithm.zhang2000.calibration import Zhang2000Calib
from algorithm.general.feature_analysis import detect_corners

# Map dynamic camera IDs to the persistent intrinsics directory names.
# Update this dictionary to point each current OpenCV camera identifier
# to the folder that holds its calibration (relative to `intrinsics_root`).
CAMERA_ID_MAP = {
    # Example:
    # "0": "front_left",
    # "1": "front_right",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a checkerboard frame and solve extrinsics using stored intrinsics."
    )
    parser.add_argument("camera_id", help="Current camera identifier (used to look up the calibration directory via CAMERA_ID_MAP).")
    parser.add_argument(
        "--camera-index",
        type=int,
        help="OpenCV capture index. Defaults to the numeric value of camera_id if possible, otherwise 0."
    )
    parser.add_argument(
        "--intrinsics-root",
        type=Path,
        default=Path("intrinsics"),
        help="Root directory that holds per-camera calibration results. [default: %(default)s]"
    )
    parser.add_argument(
        "--num-corners",
        type=int,
        nargs=2,
        metavar=("NX", "NY"),
        default=(8, 5),
        help="Checkerboard interior corner dimensions (columns, rows). [default: %(default)s]"
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=25.0,
        help="Checker square size in millimetres. [default: %(default)s]"
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional output path (.npz or .json). If a directory is provided, a timestamped file will be created."
    )
    parser.add_argument(
        "--no-preview",
        action="store_true",
        help="Disable the live preview window (capture occurs immediately)."
    )
    return parser.parse_args()


def create_checker_points(num_corners: Tuple[int, int], square_size: float) -> np.ndarray:
    nx, ny = num_corners
    world_points = np.zeros((nx * ny, 3), dtype=np.float32)
    grid = np.mgrid[0:nx, 0:ny].T.reshape(-1, 2).astype(np.float32)
    world_points[:, :2] = grid * float(square_size)
    return world_points


def format_output_path(path: Optional[Path], camera_id: str, default_suffix: str) -> Optional[Path]:
    if path is None:
        return None
    if path.suffix:
        path.parent.mkdir(parents=True, exist_ok=True)
        return path
    path.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return path / f"{camera_id}_extrinsics_{timestamp}{default_suffix}"


def solve_pose(
    frame: np.ndarray,
    checker_shape: Tuple[int, int],
    checker_size: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], float]:

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    detected, corners = detect_corners(gray, checker_shape)
    if not detected:
        return None, None, None, None, float("inf")

    corners = np.squeeze(corners).astype(np.float32)
    if corners.ndim != 2 or corners.shape[0] != checker_shape[0] * checker_shape[1]:
        return None, None, None, None, float("inf")

    object_points = create_checker_points(checker_shape, checker_size)
    h_matrix, status = cv.findHomography(object_points[:, :2], corners)
    if h_matrix is None or status is None:
        return None, None, None, None, float("inf")

    Rt, rvec, tvec = Zhang2000Calib.get_rvec_and_tvec(H=h_matrix, A=camera_matrix, idx=0)

    projection, _ = cv.projectPoints(object_points, rvec, tvec, camera_matrix, dist_coeffs)
    projection = np.squeeze(projection).astype(np.float32)
    repro_error = float(np.mean(np.linalg.norm(projection - corners, axis=1)))

    transform_checker_to_camera = np.eye(4, dtype=np.float32)
    rotation_matrix, _ = cv.Rodrigues(rvec)
    transform_checker_to_camera[:3, :3] = rotation_matrix
    transform_checker_to_camera[:3, 3] = tvec.reshape(3)

    transform_camera_to_checker = np.eye(4, dtype=np.float32)
    transform_camera_to_checker[:3, :3] = rotation_matrix.T
    transform_camera_to_checker[:3, 3] = -rotation_matrix.T @ tvec.reshape(3)

    return (
        rvec.astype(np.float32),
        tvec.astype(np.float32),
        transform_checker_to_camera,
        transform_camera_to_checker,
        repro_error,
    )


def main() -> None:
    args = parse_args()

    map_key = str(args.camera_id)
    intrinsics_folder = CAMERA_ID_MAP.get(map_key, map_key)
    intrinsics_path = args.intrinsics_root / intrinsics_folder / "calibration.txt"
    if not intrinsics_path.exists():
        raise FileNotFoundError(f"Missing intrinsics file for camera {args.camera_id!r}: {intrinsics_path}")

    calib = json.loads(intrinsics_path.read_text())
    camera_matrix = np.array(calib["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(calib["dist_coeffs"], dtype=np.float32)

    try:
        default_index = int(args.camera_id)
    except ValueError:
        default_index = 0
    camera_index = args.camera_index if args.camera_index is not None else default_index

    cap = cv.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera index {camera_index}")

    print("Press space to capture a frame for pose estimation, or 'q'/ESC to exit.")
    captured_frame: Optional[np.ndarray] = None
    checker_shape = tuple(args.num_corners)

    window_name = "Extrinsics Preview"
    if not args.no_preview:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️  Failed to read from the camera. Retrying...")
            continue

        display_frame = frame.copy()
        if not args.no_preview:
            cv.imshow(window_name, display_frame)
        key = cv.waitKey(1 if not args.no_preview else 10) & 0xFF

        if args.no_preview:
            captured_frame = frame
            break

        if key in (ord(" "), ord("c")):
            captured_frame = frame
            break
        if key in (27, ord("q")):
            cap.release()
            if not args.no_preview:
                cv.destroyWindow(window_name)
            print("Cancelled by user.")
            return

    cap.release()
    if not args.no_preview:
        cv.destroyWindow(window_name)

    if captured_frame is None:
        raise RuntimeError("No frame captured for extrinsics computation.")

    rvec, tvec, transform_checker_to_camera, transform_camera_to_checker, repro_error = solve_pose(
        captured_frame,
        checker_shape=checker_shape,
        checker_size=args.square_size,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
    )

    if rvec is None or tvec is None:
        raise RuntimeError("Checkerboard corners were not detected. Please try again with a clearer view.")

    print(f"Rotation vector (Rodrigues): {rvec.ravel()}")
    print(f"Translation vector (camera frame, mm): {tvec.ravel()}")
    print(f"Mean reprojection error: {repro_error:.4f} px")
    print("Homogeneous transform (checkerboard -> camera):")
    print(transform_checker_to_camera)
    print("Camera pose relative to checkerboard (camera -> checkerboard):")
    print(transform_camera_to_checker)

    if args.output:
        suffix = args.output.suffix if args.output.suffix else ".txt"
        output_path = format_output_path(args.output, str(args.camera_id), suffix)
        payload = {
            "camera_id": str(args.camera_id),
            "rotation_vector": rvec.ravel().tolist(),
            "translation_vector": tvec.ravel().tolist(),
            "checkerboard_to_camera_transform": transform_checker_to_camera.tolist(),
            "camera_pose_in_checkerboard": transform_camera_to_checker.tolist(),
            "reprojection_error_px": repro_error,
            "intrinsics_source": str(intrinsics_path),
        }
        output_path.write_text(json.dumps(payload, indent=2))
        print(f"Saved extrinsics to {output_path}")


if __name__ == "__main__":
    main()
