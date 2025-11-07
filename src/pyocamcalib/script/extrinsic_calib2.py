"""
Secondary CLI for estimating extrinsic parameters from a single chessboard capture.

Given a calibrated camera (JSON intrinsics), a chessboard image, and the board geometry,
the script solves for the [R|t] matrix and reports translation plus roll/pitch/yaw angles.
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
    """Detect chessboard corners with OpenCV's SB detector and refine them."""
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    for flags in (cv.CALIB_CB_NORMALIZE_IMAGE, cv.CALIB_CB_EXHAUSTIVE | cv.CALIB_CB_ACCURACY):
        ok, corners = cv.findChessboardCornersSB(gray, pattern_size, flags=flags)
        if ok:
            refined = cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), CRITERIA)
            return np.squeeze(refined, 1)[::-1]
    raise RuntimeError("Unable to detect chessboard corners in the provided image.")


def _reprojection_rms(camera: Camera,
                      world_points: np.ndarray,
                      image_points: np.ndarray,
                      extrinsic: np.ndarray) -> float:
    projected = camera.world2cam(world_points, extrinsic)
    return float(np.linalg.norm(projected - image_points, axis=1).mean())


def _estimate_extrinsic(camera: Camera,
                        image_points: np.ndarray,
                        world_points: np.ndarray,
                        image_size: Tuple[int, int]) -> Tuple[np.ndarray, float]:
    r_part, t_part = partial_extrinsics(image_points, world_points, image_size, camera.distortion_center)
    candidates = get_full_rotation_matrix(r_part, t_part, image_points, image_size, camera.distortion_center)
    errors = np.array([_reprojection_rms(camera, world_points, image_points, extrinsic)
                       for extrinsic in candidates])
    idx = int(np.argmin(errors))
    return candidates[idx], float(errors[idx])


def _rotation_matrix_to_euler(rotation: np.ndarray) -> np.ndarray:
    """Return roll, pitch, yaw (degrees) following the ZYX convention."""
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


def _draw_axes(image: np.ndarray,
               camera: Camera,
               extrinsic: np.ndarray,
               square_size: float,
               axis_length: float) -> np.ndarray:
    """Draw the chessboard origin plus X/Y axes on the original fisheye image."""
    overlay = image.copy()
    axis_extent = square_size * axis_length
    axis_points = np.array([
        [0.0, 0.0, 0.0],
        [axis_extent, 0.0, 0.0],
        [0.0, axis_extent, 0.0],
    ])
    projected = np.round(camera.world2cam(axis_points, extrinsic)).astype(int)
    origin = tuple(projected[0])
    x_axis = tuple(projected[1])
    y_axis = tuple(projected[2])
    cv.circle(overlay, origin, 6, (0, 0, 255), -1)
    cv.line(overlay, origin, x_axis, (0, 0, 255), 2)
    cv.line(overlay, origin, y_axis, (0, 255, 0), 2)
    x_label = (x_axis[0] + 5, x_axis[1] + 5)
    y_label = (y_axis[0] + 5, y_axis[1] + 5)
    cv.putText(overlay, "X", x_label, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)
    cv.putText(overlay, "Y", y_label, cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
    return overlay


def _draw_detected_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    """Overlay all detected chessboard corners."""
    overlay = image.copy()
    points = np.round(corners).astype(int)
    for idx, point in enumerate(points, start=1):
        center = tuple(point)
        cv.circle(overlay, center, 4, (255, 0, 0), -1)
        label_pos = (center[0] + 5, center[1] - 5)
        cv.putText(
            overlay,
            str(idx),
            label_pos,
            cv.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1,
            cv.LINE_AA,
        )
    return overlay

"""
python src/pyocamcalib/script/extrinsic_calib2.py ./src/pyocamcalib/checkpoints/calibration/calibration_fisheye_1_07112025_152810.json ./test_images/fish_1/Fisheye1_1.jpg 
"""

def main(
    calibration_file: Path = typer.Argument(..., help="Path to the fisheye calibration JSON file."),
    image_path: Path = typer.Argument(..., help="Path to the chessboard image."),
    chessboard_size_row: int = typer.Option(8, help="Number of inner corners along a row."),
    chessboard_size_column: int = typer.Option(6, help="Number of inner corners along a column."),
    square_size: float = typer.Option(1.0, help="Size of a chessboard square (units carry over to translation)."),
    axis_length: float = typer.Option(3.0, help="Axis length expressed in number of squares to draw."),
    output_path: Optional[Path] = typer.Option('./outputs/', help="Optional path to save the overlay image."),
):
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

    typer.echo("Extrinsic matrix [R|t]:")
    with np.printoptions(precision=6, suppress=True):
        typer.echo(extrinsic)
    typer.echo(f"Translation (units={square_size}): "
               f"x={translation[0]:.6f}, y={translation[1]:.6f}, z={translation[2]:.6f}")
    typer.echo(f"Orientation (deg): roll={roll:.3f}, pitch={pitch:.3f}, yaw={yaw:.3f}")
    typer.echo(f"Reprojection RMS error: {rms:.4f} px")

    output_file_path = Path(output_path) / f"{image_path.stem}_axes.jpg"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    overlay = _draw_axes(image, camera, extrinsic, square_size, axis_length)
    overlay = _draw_detected_corners(overlay, image_points)
    print(str(output_file_path))
    cv.imwrite(str(output_file_path), overlay)
    typer.echo(f"Overlay saved to: {output_file_path}")


if __name__ == "__main__":
    typer.run(main)
