#!/usr/bin/env python3
"""
Batch helper to generate optimized intrinsics using the Zhang 2000 calibration utilities.

For every camera id provided, the script expects checkerboard captures to live under
`data/<camera_id>/`. It runs the same pipeline implemented in `Zhang2000Calib.__call__`
to recover the optimized camera matrix and radial distortion coefficients, and stores
the results under `intrinsics/<camera_id>/calibration.txt` in a human-readable format.
"""

import argparse
import json
from pathlib import Path
from typing import Tuple

import cv2 as cv
import numpy as np

from algorithm.zhang2000.calibration import Zhang2000Calib
from algorithm.general.calib import CalibMethod


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate optimized intrinsics using the Zhang 2000 pipeline."
    )
    parser.add_argument(
        "camera_ids",
        nargs="+",
        help="Identifiers that map to folders in data/<camera_id>/."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root directory that contains per-camera image folders. [default: %(default)s]"
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("intrinsics"),
        help="Destination root where calibration results are written. [default: %(default)s]"
    )
    parser.add_argument(
        "--suffix",
        default=".jpg",
        help="Image file suffix used for calibration frames (e.g. .jpg, .png). [default: %(default)s]"
    )
    parser.add_argument(
        "--num-corners",
        type=int,
        nargs=2,
        metavar=("NX", "NY"),
        default=(13, 9),
        help="Checkerboard interior corner dimensions (columns, rows). [default: %(default)s]"
    )
    parser.add_argument(
        "--square-size",
        type=float,
        default=20.0,
        help="Checker square size in millimetres. [default: %(default)s]"
    )
    parser.add_argument(
        "--show-figure",
        action="store_true",
        help="Enable the debug figures provided by the calibration module."
    )
    parser.add_argument(
        "--include-skew",
        action="store_true",
        help="Keep the gamma term during optimization (off by default)."
    )
    parser.add_argument(
        "--skip-optimization",
        action="store_true",
        help="Skip the non-linear refinement stage and store the closed-form intrinsics instead."
    )
    return parser.parse_args()


def run_calibration(
    camera_id: str,
    data_root: Path,
    output_root: Path,
    suffix: str,
    num_corners: Tuple[int, int],
    square_size: float,
    show_figure: bool,
    include_skew: bool,
    optimize: bool
) -> None:
    data_dir = data_root / str(camera_id)
    if not data_dir.exists():
        raise FileNotFoundError(f"Missing data directory for camera {camera_id!r}: {data_dir}")

    input_files = {"img_folder": data_dir}
    config = {
        "input_file_format": suffix,
        "calibration_method": CalibMethod.ZHANG2000,
        "checkerboard": {
            "num_corners": tuple(num_corners),
            "checker_size": square_size,
            "show_figure": show_figure,
        },
        "zhang2000": {
            "get_skewness": include_skew,
            "optimize_parameters": optimize
        }
    }

    calibrator = Zhang2000Calib(input_files=input_files, config=config)

    V, H = calibrator.get_V_and_H()
    b = calibrator.get_b_vector(V)
    A = calibrator.get_intrinsic_params(b)
    alpha, beta, gamma, u0, v0 = A[0, 0], A[1, 1], A[0, 1], A[0, 2], A[1, 2]

    num_points = int(np.prod(calibrator.checker_shape))
    rvec_list = []
    tvec_list = []

    D = np.zeros((2 * num_points * calibrator.num_img_data, 2), dtype=np.float64)
    d_vec = np.zeros((2 * num_points * calibrator.num_img_data, 1), dtype=np.float64)

    for idx in range(calibrator.num_img_data):
        Rt, rvec, tvec = calibrator.get_rvec_and_tvec(H=H, A=A, idx=idx)
        rvec_list.append(rvec)
        tvec_list.append(tvec)

        projected_points2d, _ = cv.projectPoints(
            objectPoints=calibrator.points3d,
            rvec=rvec,
            tvec=tvec,
            cameraMatrix=A,
            distCoeffs=np.zeros(4, dtype=np.float32)
        )
        for j, point in enumerate(np.squeeze(projected_points2d)):
            x, y = (point - np.array([u0, v0])) / np.array([alpha, beta])
            r = x ** 2 + y ** 2

            base = 2 * idx * num_points + 2 * j
            D[base:base + 2, :] = [
                [(point[0] - u0) * r, (point[0] - u0) * r ** 2],
                [(point[1] - v0) * r, (point[1] - v0) * r ** 2],
            ]

            target_idx = np.argmin(
                np.linalg.norm(calibrator.points2d[idx] - point, axis=1)
            )
            diff = calibrator.points2d[idx][target_idx] - point
            d_vec[base:base + 2, 0] = diff

    k = np.linalg.pinv(D.T @ D) @ D.T @ d_vec
    k1, k2 = k.flatten()
    dist_coeffs = np.array([k1, k2, 0.0, 0.0], dtype=np.float32)

    if optimize:
        updated_params, num_intrinsic_params = calibrator.optimize_params(
            A=A, k1=k1, k2=k2, rvec_list=rvec_list, tvec_list=tvec_list
        )

        if include_skew:
            camera_matrix = np.array(
                [
                    [updated_params[0], updated_params[2], updated_params[3]],
                    [0.0, updated_params[1], updated_params[4]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            dist_coeffs = np.array(
                [updated_params[5], updated_params[6], 0.0, 0.0],
                dtype=np.float32,
            )
        else:
            camera_matrix = np.array(
                [
                    [updated_params[0], 0.0, updated_params[2]],
                    [0.0, updated_params[1], updated_params[3]],
                    [0.0, 0.0, 1.0],
                ],
                dtype=np.float32,
            )
            dist_coeffs = np.array(
                [updated_params[4], updated_params[5], 0.0, 0.0],
                dtype=np.float32,
            )

        rvec_array = np.array(updated_params[num_intrinsic_params:
                                             num_intrinsic_params + 3 * calibrator.num_img_data])
        tvec_array = np.array(updated_params[num_intrinsic_params + 3 * calibrator.num_img_data:])
        rvecs = rvec_array.reshape(calibrator.num_img_data, 3)
        tvecs = tvec_array.reshape(calibrator.num_img_data, 3)
    else:
        camera_matrix = A.astype(np.float32)
        rvecs = np.vstack([r.reshape(1, 3) for r in rvec_list])
        tvecs = np.vstack([t.reshape(1, 3) for t in tvec_list])

    repro_errors = []
    for idx in range(calibrator.num_img_data):
        projected_points2d, _ = cv.projectPoints(
            objectPoints=calibrator.points3d,
            rvec=rvecs[idx],
            tvec=tvecs[idx],
            cameraMatrix=camera_matrix,
            distCoeffs=dist_coeffs,
        )
        repro_errors.append(
            calibrator.calculate_reprojection_error(
                calibrator.points2d[idx], projected_points2d
            )
        )

    output_dir = output_root / str(camera_id)
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "camera_id": str(camera_id),
        "camera_matrix": camera_matrix.tolist(),
        "dist_coeffs": dist_coeffs.tolist(),
        "rvecs": rvecs.astype(np.float32).tolist(),
        "tvecs": tvecs.astype(np.float32).tolist(),
        "reprojection_errors_px": np.array(repro_errors, dtype=np.float32).tolist(),
        "num_images": calibrator.num_img_data,
        "mean_reprojection_error": float(np.mean(repro_errors)),
        "checkerboard": {
            "num_corners": list(num_corners),
            "square_size_mm": square_size,
        },
        "optimize": bool(optimize),
        "include_skew": bool(include_skew),
        "image_suffix": suffix,
        "data_source": str(data_dir),
    }
    (output_dir / "calibration.txt").write_text(json.dumps(payload, indent=2))

    print(
        f"[{camera_id}] "
        f"mean reprojection error: {payload['mean_reprojection_error']:.4f} px | "
        f"saved to {output_dir}"
    )


def main() -> None:
    args = parse_args()
    for camera_id in args.camera_ids:
        run_calibration(
            camera_id=camera_id,
            data_root=args.data_root,
            output_root=args.output_root,
            suffix=args.suffix,
            num_corners=tuple(args.num_corners),
            square_size=args.square_size,
            show_figure=args.show_figure,
            include_skew=args.include_skew,
            optimize=not args.skip_optimization,
        )


if __name__ == "__main__":
    main()
