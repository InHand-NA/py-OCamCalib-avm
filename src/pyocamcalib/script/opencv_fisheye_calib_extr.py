"""
OpenCV fisheye extrinsics calibration CLI for a single chessboard image.

Loads fisheye intrinsics (K, D), detects chessboard corners, undistorts to
normalized coordinates, solves PnP (IPPE preferred), and saves/visualizes
the pose. Projects axes using cv2.fisheye.projectPoints.
"""
from pathlib import Path
from typing import Optional, Tuple
import json

import cv2 as cv
import numpy as np
import typer


def _load_intrinsics(path: Path):
    with open(path, "r") as f:
        data = json.load(f)
    K = np.array(data["K"], dtype=np.float64)
    D = np.array(data["D"], dtype=np.float64).reshape(-1, 1)
    img_size = tuple(data.get("image_size", [0, 0]))
    return K, D, img_size


def _build_object_points(pattern_size: Tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 1, 3), np.float64)
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    objp[:, 0, 0] = grid_x.reshape(-1) * square_size
    objp[:, 0, 1] = grid_y.reshape(-1) * square_size
    objp[:, 0, 2] = 0.0
    return objp


def _detect_corners(image: np.ndarray, pattern_size: Tuple[int, int]) -> Optional[np.ndarray]:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    if hasattr(cv, "findChessboardCornersSB"):
        ok, corners = cv.findChessboardCornersSB(gray, pattern_size)
        if not ok:
            flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
            ok, corners = cv.findChessboardCorners(gray, pattern_size, flags)
    else:
        flags = cv.CALIB_CB_ADAPTIVE_THRESH | cv.CALIB_CB_NORMALIZE_IMAGE
        ok, corners = cv.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        return None
    term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
    return corners.reshape(-1, 1, 2)


def _solve_extrinsics_fisheye(K: np.ndarray,
                              D: np.ndarray,
                              img_points: np.ndarray,
                              obj_points: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    # Undistort to normalized coordinates
    und = cv.fisheye.undistortPoints(img_points, K, D)  # (N,1,2) normalized
    uv = und.reshape(-1, 2)
    # PnP with K=I, no distortion
    # Prefer IPPE for planar targets, fall back to ITERATIVE on failure
    ok, rvec, tvec = cv.solvePnP(obj_points, uv, np.eye(3), None, flags=cv.SOLVEPNP_IPPE)
    if not ok:
        ok, rvec, tvec = cv.solvePnP(obj_points, uv, np.eye(3), None, flags=cv.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("solvePnP failed")
    return rvec, tvec


def _draw_axes(image: np.ndarray,
               K: np.ndarray,
               D: np.ndarray,
               rvec: np.ndarray,
               tvec: np.ndarray,
               square_size: float,
               axis_length: float) -> np.ndarray:
    overlay = image.copy()
    L = square_size * axis_length
    axis = np.array([
        [[0.0, 0.0, 0.0]],
        [[L, 0.0, 0.0]],
        [[0.0, L, 0.0]],
    ], dtype=np.float64)
    proj, _ = cv.fisheye.projectPoints(axis, rvec, tvec, K, D)
    proj = proj.reshape(-1, 2)
    o = tuple(np.round(proj[0]).astype(int))
    x = tuple(np.round(proj[1]).astype(int))
    y = tuple(np.round(proj[2]).astype(int))
    cv.circle(overlay, o, 6, (0, 0, 255), -1)
    cv.line(overlay, o, x, (0, 0, 255), 2)
    cv.line(overlay, o, y, (0, 255, 0), 2)
    cv.putText(overlay, "X", (x[0] + 5, x[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)
    cv.putText(overlay, "Y", (y[0] + 5, y[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
    return overlay

# CLI examples:
# 1) Using intrinsics JSON from the intrinsics script:
#    python -m pyocamcalib.script.opencv_fisheye_calib_extr \
#        ./src/pyocamcalib/checkpoints/calibration/opencv_fisheye_intrinsics_fe1.json \
#        ./test_images/fe1_1.jpg --chessboard-size-row 9 --chessboard-size-column 6 \
#        --square-size 30 --axis-length 5 --output ./outputs

def main(
    intrinsics_json: Path = typer.Argument(..., help="Path to intrinsics JSON from opencv_fisheye_calib_intr.py"),
    image_path: Path = typer.Argument(..., help="Path to a chessboard image"),
    chessboard_size_row: int = typer.Option(9, help="Number of inner corners along a row (columns)."),
    chessboard_size_column: int = typer.Option(6, help="Number of inner corners along a column (rows)."),
    square_size: float = typer.Option(30.0, help="Chessboard square size (units cm/mm/etc.)"),
    axis_length: float = typer.Option(5.0, help="Axis length in squares for visualization."),
    output: Optional[Path] = typer.Option("./outputs/", help="Folder to save overlay image and JSON pose."),
):
    if not intrinsics_json.is_file():
        raise typer.BadParameter(f"Intrinsics file not found: {intrinsics_json}")
    if not image_path.is_file():
        raise typer.BadParameter(f"Image file not found: {image_path}")

    K, D, _ = _load_intrinsics(intrinsics_json)
    img = cv.imread(str(image_path))
    if img is None:
        raise typer.BadParameter(f"Unable to read image: {image_path}")

    pattern_size = (chessboard_size_row, chessboard_size_column)
    corners = _detect_corners(img, pattern_size)
    if corners is None:
        raise RuntimeError("Chessboard detection failed")

    objp = _build_object_points(pattern_size, square_size)
    rvec, tvec = _solve_extrinsics_fisheye(K, D, corners, objp)

    # Compute simple RMS reprojection error in pixel domain
    proj, _ = cv.fisheye.projectPoints(objp, rvec, tvec, K, D)
    err = np.linalg.norm(proj.reshape(-1, 2) - corners.reshape(-1, 2), axis=1)
    rms = float(err.mean())

    # Save overlay and pose
    out_dir = Path(output)
    out_dir.mkdir(parents=True, exist_ok=True)
    overlay = _draw_axes(img, K, D, rvec, tvec, square_size, axis_length)
    overlay_path = out_dir / f"{image_path.stem}_fisheye_axes.jpg"
    cv.imwrite(str(overlay_path), overlay)

    pose_json = out_dir / f"{image_path.stem}_fisheye_extrinsics.json"
    out = {
        "rvec": rvec.reshape(-1).tolist(),
        "tvec": tvec.reshape(-1).tolist(),
        "rms_px": rms,
    }
    with open(pose_json, "w") as f:
        json.dump(out, f, indent=2)

    # Report
    with np.printoptions(precision=6, suppress=True):
        typer.echo(f"rvec = {rvec.reshape(-1)}")
        typer.echo(f"tvec = {tvec.reshape(-1)}")
    typer.echo(f"Reprojection RMS: {rms:.4f} px")
    typer.echo(f"Overlay saved: {overlay_path}")
    typer.echo(f"Pose saved: {pose_json}")


if __name__ == "__main__":
    typer.run(main)
