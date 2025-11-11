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
from tqdm import tqdm
from itertools import product
import json
import glob
import math
from dataclasses import dataclass
from typing import Tuple, List, Dict

from scipy.optimize import least_squares
from loguru import logger

from pyocamcalib.core.extrinsic import get_full_rotation_matrix, partial_extrinsics
from pyocamcalib.modelling.camera import Camera
from pyocamcalib.modelling.utils import generate_checkerboard_points
from pyocamcalib.core._utils import get_reprojection_error_all, get_reprojection_error
from pyocamcalib.core.linear_estimation import get_first_linear_estimate, get_taylor_linear
from pyocamcalib.core.optim import bundle_adjustement
from pyocamcalib.modelling.utils import get_files, generate_checkerboard_points, check_detection, transform, save_calib, \
    get_canonical_projection_model, Loader, get_incident_angle

CRITERIA = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 40, 1e-3)

# ----------------------------
# Geometry helpers
# ----------------------------
def rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv.Rodrigues(rvec.astype(np.float64))
    return R


def project_points_ocam(Pw: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, ocam: Camera) -> np.ndarray:
    """
    Pw: (N,3) points in board/world frame (e.g., Z=0 if chessboard plane)
    rvec: (3,), tvec: (3,)
    returns pixels (N,2)
    """
    R = rodrigues_to_R(rvec)
    Xc = (Pw @ R.T) + tvec[None, :]
    # convert to unit direction
    Xc_norm = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-16)
    uv = ocam.world2cam(Xc_norm, None)
    return uv


def residuals_pose(params: np.ndarray, Pw: np.ndarray, uv_obs: np.ndarray, ocam: Camera) -> np.ndarray:
    rvec = params[0:3]
    tvec = params[3:6]
    uv_pred = project_points_ocam(Pw, rvec, tvec, ocam)
    return (uv_pred - uv_obs).ravel()


def solve_pose_ocam(Pw: np.ndarray, uv: np.ndarray, ocam: Camera,
                    rvec0: np.ndarray = None, tvec0: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Nonlinear least squares on OCam projection to estimate rvec,tvec.
    Returns rvec, tvec, rms_error (pixels).
    """
    if rvec0 is None:
        rvec0 = np.array([0.0, 0.0, 0.0])
    if tvec0 is None:
        # rough depth guess: 1m along +Z of camera looking at board
        tvec0 = np.array([0.0, 0.0, 1.0])

    x0 = np.hstack([rvec0, tvec0])
    res = least_squares(
        residuals_pose, x0,
        args=(Pw, uv, ocam),
        method="lm", max_nfev=200,
        xtol=1e-10, ftol=1e-10, gtol=1e-10
    )
    rvec = res.x[0:3]
    tvec = res.x[3:6]
    rms = math.sqrt(np.mean(res.fun**2))
    return rvec, tvec, rms


class ExtCalibrationEngine:
    def __init__(self,
                 working_dir: str,
                 chessboard_size: Tuple[int, int],
                 camera_name: str,
                 square_size: float = 1):
        """
        :param working_dir: path to folder which contains all chessboard images
        :param chessboard_size: Number of INNER corners per a chessboard (row, column)
        """
        self.rms_std_list = None
        self.rms_mean_list = None
        self.rms_overall = None
        self.extrinsics_t_linear = None
        self.taylor_coefficient_linear = None
        self.working_dir = Path(working_dir)
        self.images_path = [str(e) for e in get_files(Path(working_dir))]
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.sensor_size = cv.imread(str(self.images_path[0])).shape[:2][::-1]
        self.distortion_center = (self.sensor_size[0] / 2, self.sensor_size[1] / 2)
        self.detections = {}
        self.image_points = None
        self.world_points = None
        self.distortion_center_linear = None
        self.extrinsics_t = None
        self.taylor_coefficient = None
        self.stretch_matrix = None
        self.valid_pattern = None
        self.cam_name = camera_name
        self.inverse_poly = None
        pass


    def generate_checkerboard_points(self, z_axis=True):
        # get 3D checkerboard points
        pass


    def detect_corners(self, images_file_path, check: bool = False, max_height: int = 520):
        images_path = [images_file_path]
        count = 0
        world_points = generate_checkerboard_points(self.chessboard_size, self.square_size, z_axis=True)

        logger.info(f"Start corners extraction at {images_file_path}, desired chessboard size {self.chessboard_size}")

        for img_f in tqdm(sorted(images_path)):
            print(f"detect on image: {img_f}")
            img = cv.imread(str(img_f))
            height, width = img.shape[:2]
            ratio = width / height
            img_resize = cv.resize(img, (round(ratio * max_height), max_height))
            r_h = height / max_height
            r_w = width / (ratio * max_height)

            print(f"h: {height}; w: {width}; ratio: {ratio}")

            gray_resize = cv.cvtColor(img_resize, cv.COLOR_BGR2GRAY)
            gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
            for block, bias in list(product(range(20, 40, 5), range(-10, 31, 5))):

                block = (block // 2) * 2 + 1
                img_bw = cv.adaptiveThreshold(gray_resize, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY, block,
                                              bias)
                cv.imshow('img bw', img_bw)
                ret, corners = cv.findChessboardCornersSB(img_bw, self.chessboard_size, flags=cv.CALIB_CB_EXHAUSTIVE)
                print(f"find corners 1: {corners}")
                if not ret:
                    ret, corners = cv.findChessboardCornersSB(img_bw, self.chessboard_size, flags=0)
                print(f"find corners 2: {corners}")
                if ret:
                    corners = np.squeeze(corners)
                    corners[:, 0] *= r_w
                    corners[:, 1] *= r_h
                    win_size = (5, 5)
                    zero_zone = (-1, -1)
                    criteria = (cv.TERM_CRITERIA_EPS + cv.TermCriteria_COUNT, 40, 0.001)
                    corners = np.expand_dims(corners, axis=0)
                    cv.cornerSubPix(gray, corners, win_size, zero_zone, criteria)
                    if check:
                        check_detection(np.squeeze(corners), img)
                    count += 1
                    self.detections[str(img_f)] = {"image_points": np.squeeze(corners)[::-1],
                                                   "world_points": np.squeeze(world_points)}
                    self.image_points = np.squeeze(corners)[::-1]
                    self.world_points = np.squeeze(world_points)
                    break

        logger.info(f"Extracted chessboard corners with success = {count}/{len(images_path)}")

    def extract_extrinsic(self, ocam, visualize=True):
        pass

    def visualize(self):
        pass

"""
python src/pyocamcalib/script/extrinsic_calib2.py ./src/pyocamcalib/checkpoints/calibration/calibration_inhandus_1_10112025_113611.json ./test_images/inhandus_1/fe1_3.jpg 
"""

def main(
    calibration_file: Path = typer.Argument(..., help="Path to the fisheye calibration JSON file."),
    image_path: Path = typer.Argument(..., help="Path to the chessboard image."),
    chessboard_size_row: int = typer.Option(8, help="Number of inner corners along a row."),
    chessboard_size_column: int = typer.Option(6, help="Number of inner corners along a column."),
    square_size: float = typer.Option(32.5, help="Size of a chessboard square (units carry over to translation)."),
    axis_length: float = typer.Option(65.0, help="Axis length expressed in number of squares to draw."),
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

    working_dir = "./"
    camera_name = "inhandus_1"
    image = cv.imread(str(image_path))
    if image is None:
        raise typer.BadParameter(f"Unable to read image: {image_path}")

    camera = Camera.load_parameters_json(str(calibration_file))
    pattern_size = (chessboard_size_row, chessboard_size_column)

    chessboard_size = (chessboard_size_row, chessboard_size_column)
    my_calib_engine = ExtCalibrationEngine(working_dir, chessboard_size, camera_name, square_size)
    my_calib_engine.detect_corners(image_path, check=True, max_height=520)
    pass


if __name__ == "__main__":
    typer.run(main)
