#!/usr/bin/env python3
"""
Interactive extrinsics solver that reuses the Zhang 2000 helpers.

Camera sources are defined in CAMERA_ID_MAP, which maps the dynamic capture ID to the
intrinsics subdirectory containing `calibration.txt`. For each entry, the script streams
frames, captures on demand, and reports the camera pose relative to the checkerboard,
saving both a JSON summary and human-readable outputs.
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple

import cv2 as cv
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from algorithm.zhang2000.calibration import Zhang2000Calib
from algorithm.general.feature_analysis import detect_corners

# Map dynamic camera IDs to the persistent intrinsics directory names.
# Update this dictionary to point each current OpenCV camera identifier
# to the folder that holds its calibration (relative to `intrinsics_root`).
CAMERA_ID_MAP = {
    # Example:
    # "0": "front_left",
    # "1": "front_right",
    "4": "front_head"
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a checkerboard frame and solve extrinsics using stored intrinsics."
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
        default=(12, 8),
        help="Checkerboard interior corner dimensions (columns, rows). [default: %(default)s]"
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=20.0,
        help="Checker square size in millimetres. [default: %(default)s]"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("extrinsics"),
        help="Directory where per-camera results will be stored. [default: %(default)s]"
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


def rotation_matrix_to_rpy(R: np.ndarray) -> Tuple[float, float, float]:
    """Convert rotation matrix to roll, pitch, yaw (X, Y, Z) using the XYZ convention."""
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6

    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0

    return float(roll), float(pitch), float(yaw)


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


def process_camera(camera_key: str, intrinsics_folder: str, args: argparse.Namespace) -> None:
    print(f"\n=== Processing camera {camera_key} (intrinsics: {intrinsics_folder}) ===")

    intrinsics_path = args.intrinsics_root / intrinsics_folder / "calibration.txt"
    if not intrinsics_path.exists():
        raise FileNotFoundError(
            f"Missing intrinsics file for camera {camera_key!r}: {intrinsics_path}"
        )

    calib = json.loads(intrinsics_path.read_text())
    camera_matrix = np.array(calib["camera_matrix"], dtype=np.float32)
    dist_coeffs = np.array(calib["dist_coeffs"], dtype=np.float32)

    try:
        camera_index: Optional[int] = int(camera_key)
    except ValueError:
        camera_index = None

    capture_source = camera_index if camera_index is not None else camera_key
    cap = cv.VideoCapture(capture_source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open camera source {capture_source!r}")

    print("Press space to capture a frame for pose estimation, or 'q'/ESC to skip this camera.")
    captured_frame: Optional[np.ndarray] = None
    checker_shape = tuple(args.num_corners)

    window_name = f"Extrinsics Preview - {camera_key}"
    if not args.no_preview:
        cv.namedWindow(window_name, cv.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("⚠️  Failed to read from the camera. Retrying...")
                continue

            if not args.no_preview:
                cv.imshow(window_name, frame)
            key = cv.waitKey(1 if not args.no_preview else 10) & 0xFF

            if args.no_preview:
                captured_frame = frame
                break

            if key in (ord(" "), ord("c")):
                captured_frame = frame
                break
            if key in (27, ord("q")):
                print("Skipped by user.")
                return
    finally:
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

    translation_mm = tvec.reshape(-1)
    rotation_rad = rvec.reshape(-1)
    pose_lines = [
        f"=== Camera {camera_key} Extrinsics ===",
        f"Camera {camera_key} translation (mm):",
        f"  tx (X forward): {translation_mm[0]:.6f}",
        f"  ty (Y left): {translation_mm[1]:.6f}",
        f"  tz (Z up): {translation_mm[2]:.6f}",
        "Camera rotation (Rodrigues, rad):",
        f"  rx (roll about X): {rotation_rad[0]:.6f}",
        f"  ry (pitch about Y): {rotation_rad[1]:.6f}",
        f"  rz (yaw about Z): {rotation_rad[2]:.6f}",
    ]
    print("\n".join(pose_lines))

    output_base = args.output_root / intrinsics_folder
    output_base.mkdir(parents=True, exist_ok=True)
    standard_pose_path = output_base / "standard_pose.json"
    adjustments_lines: list[str] = []
    delta_translation = None
    delta_rpy = None

    current_pose_payload = {
        "camera_id": str(camera_key),
        "camera_index": camera_index if camera_index is not None else camera_key,
        "intrinsics_folder": intrinsics_folder,
        "rotation_vector": rvec.ravel().tolist(),
        "translation_vector": tvec.ravel().tolist(),
        "checkerboard_to_camera_transform": transform_checker_to_camera.tolist(),
        "camera_pose_in_checkerboard": transform_camera_to_checker.tolist(),
        "reprojection_error_px": repro_error,
        "intrinsics_source": str(intrinsics_path),
    }

    if standard_pose_path.exists():
        standard_data = json.loads(standard_pose_path.read_text())
        if "checkerboard_to_camera_transform" in standard_data:
            standard_checker_to_camera = np.array(
                standard_data["checkerboard_to_camera_transform"], dtype=np.float64
            )
            offset = transform_camera_to_checker.astype(np.float64) @ standard_checker_to_camera
            delta_translation = offset[:3, 3]
            roll, pitch, yaw = rotation_matrix_to_rpy(offset[:3, :3])
            delta_rpy = (roll, pitch, yaw)
            adjustments_lines = [
                "==== Adjustments to reach standard pose: =====",
                f"  Δx (mm): {delta_translation[0]:.6f}",
                f"  Δy (mm): {delta_translation[1]:.6f}",
                f"  Δz (mm): {delta_translation[2]:.6f}",
                f"  Pitch (Y, rad): {pitch:.6f}",
                f"  Roll (X, rad): {roll:.6f}",
                f"  Yaw (Z, rad): {yaw:.6f}",
            ]
            print("\n".join(adjustments_lines))
        else:
            print("⚠️  Standard pose file is missing transforms; overwriting with current pose.")
            standard_pose_path.write_text(
                json.dumps(
                    {
                        **current_pose_payload,
                        "created_at": datetime.now().isoformat(timespec="seconds"),
                    },
                    indent=2,
                )
            )
            print(f"Standard pose refreshed at {standard_pose_path}")
    else:
        standard_payload = {
            **current_pose_payload,
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }
        standard_pose_path.write_text(json.dumps(standard_payload, indent=2))
        print(f"Stored current pose as standard pose at {standard_pose_path}")

    output_path = format_output_path(output_base, intrinsics_folder, ".txt")
    image_path = output_path.with_name(output_path.stem + "_frame.png")
    pose_txt_path = output_path.parent / "camera_pose.txt"

    payload = {
        **current_pose_payload,
        "captured_image_path": str(image_path),
        "camera_pose_file": str(pose_txt_path),
        "standard_pose_file": str(standard_pose_path),
    }
    if delta_translation is not None and delta_rpy is not None:
        payload["delta_to_standard"] = {
            "translation_mm": delta_translation.tolist(),
            "rotation_rpy_rad": {
                "roll": delta_rpy[0],
                "pitch": delta_rpy[1],
                "yaw": delta_rpy[2],
            },
        }
    output_path.write_text(json.dumps(payload, indent=2))
    if not cv.imwrite(str(image_path), captured_frame):
        print(f"⚠️  Failed to save captured image to {image_path}")
    pose_txt_contents = pose_lines[:]
    if adjustments_lines:
        pose_txt_contents.append("")
        pose_txt_contents.extend(adjustments_lines)
    pose_txt_path.write_text("\n".join(pose_txt_contents) + "\n")
    print(f"Saved extrinsics to {output_path}")
    print(f"Saved captured frame to {image_path}")
    print(f"Saved pose summary to {pose_txt_path}")


def main() -> None:
    args = parse_args()

    if not CAMERA_ID_MAP:
        raise ValueError("CAMERA_ID_MAP is empty. Populate it with camera-id to intrinsics-folder mappings.")

    for camera_key, intrinsics_folder in CAMERA_ID_MAP.items():
        process_camera(camera_key, intrinsics_folder, args)


if __name__ == "__main__":
    main()
