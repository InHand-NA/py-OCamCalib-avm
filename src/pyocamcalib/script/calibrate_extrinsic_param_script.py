"""
CLI utility that estimates the extrinsic parameters of a single chessboard capture using an
existing fisheye calibration. The script loads the JSON intrinsics, detects the chessboard corners,
solves for the [R|t] matrix, draws the chessboard frame (origin plus x/y axes) on the image,
and prints translation together with roll/pitch/yaw angles.
"""
from pathlib import Path
from typing import Optional, Tuple

import cv2 as cv
import numpy as np
import typer

from pyocamcalib.core.extrinsic import get_full_rotation_matrix, partial_extrinsics
from pyocamcalib.modelling.camera import Camera
from pyocamcalib.modelling.utils import generate_checkerboard_points


CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 1e-3)


def _detect_corners(image: np.ndarray, pattern_size: Tuple[int, int]) -> np.ndarray:
    """Detect chessboard corners with a couple of fallbacks for robustness."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for flags in (cv.CALIB_CB_NORMALIZE_IMAGE, cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY):
        ok, corners = cv.findChessboardCornersSB(gray, pattern_size, flags=flags)
        if ok:
            refined = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), CRITERIA)
            # Reverse ordering to match the world point layout used during calibration.
            return np.squeeze(refined, axis=1)[::-1]
    raise RuntimeError("Unable to locate chessboard corners in the provided image.")


def _reprojection_rms(camera: Camera,
                      world_points: np.ndarray,
                      image_points: np.ndarray,
                      extrinsic: np.ndarray) -> float:
    """Compute RMS reprojection error to pick the best extrinsic hypothesis."""
    projected = camera.world2cam(world_points, extrinsic)
    return float(np.linalg.norm(projected - image_points, axis=1).mean())


def _estimate_extrinsic(camera: Camera,
                        image_points: np.ndarray,
                        world_points: np.ndarray,
                        image_size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    """Estimate [R|t] and return the best candidate together with its RMS error."""
    r_part, t_part = partial_extrinsics(image_points, world_points, image_size, camera.distortion_center)
    candidates = get_full_rotation_matrix(r_part, t_part, image_points, image_size, camera.distortion_center)
    errors = np.array([_reprojection_rms(camera, world_points, image_points, extrinsic)
                       for extrinsic in candidates])
    best_idx = int(np.argmin(errors))
    return candidates[best_idx], float(errors[best_idx])


def _rotation_matrix_to_euler(rotation: np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw (degrees) using ZYX convention."""
    sy = np.sqrt(rotation[0, 0] ** 2 + rotation[1, 0] ** 2)
    singular = sy < 1e-9
    if not singular:
        roll = np.arctan2(rotation[2, 1], rotation[2, 2])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    else:
        roll = np.arctan2(-rotation[1, 2], rotation[1, 1])
        pitch = np.arctan2(-rotation[2, 0], sy)
        yaw = 0.0
    return np.degrees([roll, pitch, yaw])


def _project_points_perspective(points: np.ndarray,
                                extrinsic: np.ndarray,
                                focal_length: float,
                                sensor_size: Tuple[int, int]) -> np.ndarray:
    """Project 3D points with a pinhole camera defined by ``focal_length`` and ``sensor_size``."""
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    camera_points = points @ rotation.T + translation
    z = camera_points[:, 2]
    if np.any(z <= 1e-9):
        raise RuntimeError("Cannot project points located behind or on the camera plane.")
    normalized = camera_points[:, :2] / z[:, None]
    cx = sensor_size[1] / 2.0
    cy = sensor_size[0] / 2.0
    projected = np.empty_like(normalized)
    projected[:, 0] = normalized[:, 0] * focal_length + cx
    projected[:, 1] = normalized[:, 1] * focal_length + cy
    return projected


def _draw_axes(image: np.ndarray,
               camera: Camera,
               extrinsic: np.ndarray,
               square_size: float,
               axis_length: float,
               perspective: Optional[Tuple[float, Tuple[int, int]]] = None) -> np.ndarray:
    """Overlay origin plus x/y axes on top of the input image."""
    overlay = image.copy()
    axis_extent = square_size * axis_length
    axis_points = np.array([
        [0.0, 0.0, 0.0],
        [axis_extent, 0.0, 0.0],
        [0.0, axis_extent, 0.0],
    ])
    if perspective is None:
        projected = np.round(camera.world2cam(axis_points, extrinsic)).astype(int)
    else:
        focal_length, sensor_size = perspective
        projected = np.round(_project_points_perspective(axis_points, extrinsic, focal_length, sensor_size)).astype(int)
        width = sensor_size[1]
        height = sensor_size[0]
        projected[:, 0] = np.clip(projected[:, 0], 0, width - 1)
        projected[:, 1] = np.clip(projected[:, 1], 0, height - 1)
    origin = tuple(projected[0])
    cv.circle(overlay, origin, 6, (0, 0, 255), -1)
    cv.line(overlay, origin, tuple(projected[1]), (0, 0, 255), 2)
    cv.line(overlay, origin, tuple(projected[2]), (0, 255, 0), 2)
    return overlay


def _estimate_fov(camera: Camera, sensor_size: Tuple[int, int], samples: int = 64) -> float:
    """Estimate the usable horizontal FOV by sampling the border of the fisheye frame."""
    height, width = sensor_size
    xs = np.linspace(0, width - 1, samples)
    ys = np.linspace(0, height - 1, samples)
    top = np.column_stack((xs, np.zeros_like(xs)))
    bottom = np.column_stack((xs, np.full_like(xs, height - 1)))
    left = np.column_stack((np.zeros_like(ys), ys))
    right = np.column_stack((np.full_like(ys, width - 1), ys))
    border_points = np.vstack((top, bottom, left, right)).astype(np.float64)
    rays = camera.cam2world(border_points)
    valid = np.isfinite(rays).all(axis=1)
    if not np.any(valid):
        return 120.0
    cos_theta = np.clip(rays[valid, 2], -1.0, 1.0)
    theta_max = float(np.max(np.arccos(cos_theta)))
    theta_max = max(theta_max, np.deg2rad(0.5))
    fov = np.degrees(2.0 * theta_max)
    return float(np.clip(fov, 1.0, 179.0))


def _perspective_focal_length(fov: float, sensor_size: Tuple[int, int]) -> float:
    """Return focal length (pixels) matching ``fov`` for an image with ``sensor_size``."""
    return float(np.max(sensor_size) / (2.0 * np.tan(np.deg2rad(fov / 2.0))))


def main(
    calibration_file: Path = typer.Argument(..., help="Path to the fisheye calibration JSON file."),
    image_path: Path = typer.Argument(..., help="Chessboard image to process."),
    chessboard_size_row: int = typer.Option(..., help="Number of inner corners along a row."),
    chessboard_size_column: int = typer.Option(..., help="Number of inner corners along a column."),
    square_size: float = typer.Option(1.0, help="Chessboard square size (units carry over to translation output)."),
    axis_length: float = typer.Option(3.0, help="Axis length expressed in number of squares to draw."),
    output_path: Optional[Path] = typer.Option(None, help="Optional path to save the overlay image."),
):
    """
    Example:
        python -m pyocamcalib.script.calibrate_extrinsic_param_script calibration.json board.jpg \\
            --chessboard-size-row 8 --chessboard-size-column 11 --square-size 30

        python src/pyocamcalib/script/calibrate_extrinsic_param_script.py src/pyocamcalib/checkpoints/calibration/calibration_fisheye_1_07112025_152810.json  ./test_images/fish_1/Fisheye1_1.jpg --chessboard-size-row 8 --chessboard-size-column 6 --square-size 32.5
    """
    if not calibration_file.is_file():
        raise typer.BadParameter(f"Calibration file not found: {calibration_file}")
    if not image_path.is_file():
        raise typer.BadParameter(f"Image file not found: {image_path}")
    if chessboard_size_row <= 1 or chessboard_size_column <= 1:
        raise typer.BadParameter("Chessboard dimensions must be greater than one.")
    if square_size <= 0 or axis_length <= 0:
        raise typer.BadParameter("Square size and axis length must be positive numbers.")

    image = cv.imread(str(image_path))
    if image is None:
        raise typer.BadParameter(f"Unable to read image: {image_path}")

    camera = Camera.load_parameters_json(str(calibration_file))
    pattern_size = (chessboard_size_row, chessboard_size_column)
    image_points = _detect_corners(image, pattern_size)
    world_points = generate_checkerboard_points(pattern_size, square_size, z_axis=True)

    extrinsic, rms = _estimate_extrinsic(camera, image_points, world_points, image.shape[:2])
    rotation = extrinsic[:, :3]
    translation = extrinsic[:, 3]
    roll, pitch, yaw = _rotation_matrix_to_euler(rotation)
    sensor_size = image.shape[:2]
    fov = _estimate_fov(camera, sensor_size)
    focal_length = _perspective_focal_length(fov, sensor_size)
    undistorted = camera.cam2perspective_indirect(image, fov, sensor_size)
    overlay = _draw_axes(
        undistorted,
        camera,
        extrinsic,
        square_size,
        axis_length,
        perspective=(focal_length, sensor_size),
    )
    if output_path is None:
        output_path = Path("outputs") / f"{image_path.stem}_axes.png"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_path), overlay)

    typer.echo("Extrinsic matrix [R|t]:")
    with np.printoptions(precision=6, suppress=True):
        typer.echo(extrinsic)
    typer.echo(f"Translation (units={square_size}): "
               f"x={translation[0]:.6f}, y={translation[1]:.6f}, z={translation[2]:.6f}")
    typer.echo(f"Orientation (deg): roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
    typer.echo(f"Reprojection RMS error: {rms:.4f} px")
    typer.echo(f"Overlay saved to: {output_path}")


if __name__ == "__main__":
    typer.run(main)
