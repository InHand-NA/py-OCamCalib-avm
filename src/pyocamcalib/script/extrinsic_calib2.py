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
        self.image = None
        self.image_path = None
        self.distortion_center_linear = None
        self.extrinsics_t = None
        self.taylor_coefficient = None
        self.stretch_matrix = None
        self.valid_pattern = None
        self.cam_name = camera_name
        self.inverse_poly = None
        pass


    def generate_checkerboard_points(self, z_axis: bool = True) -> np.ndarray:
        """生成棋盘格世界坐标并缓存。"""
        pts = generate_checkerboard_points(self.chessboard_size, self.square_size, z_axis=z_axis)
        self.world_points = np.squeeze(pts)
        return self.world_points


    def detect_corners(self, images_file_path, check: bool = False, max_height: int = 520):
        images_path = [images_file_path]
        count = 0
        world_points = generate_checkerboard_points(self.chessboard_size, self.square_size, z_axis=True)

        logger.info(f"Start corners extraction at {images_file_path}, desired chessboard size {self.chessboard_size}")

        for img_f in tqdm(sorted(images_path)):
            print(f"detect on image: {img_f}")
            self.image_path = str(img_f)
            img = cv.imread(self.image_path)
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
                    self.detections[self.image_path] = {"image_points": np.squeeze(corners)[::-1],
                                                        "world_points": np.squeeze(world_points)}
                    self.image_points = np.squeeze(corners)[::-1]
                    self.world_points = np.squeeze(world_points)
                    self.image = img
                    break

        logger.info(f"Extracted chessboard corners with success = {count}/{len(images_path)}")

    def extract_extrinsic(self,
                          camera: Camera,
                          depth_prior: Optional[float] = None,
                          depth_weight: float = 0.0) -> Tuple[np.ndarray, float]:
        """方法1改进：线性候选 + OCam 非线性细化，返回最优 [R|t] 与像素域误差。

        可选加入弱先验：在 LM 残差向量中附加 sqrt(depth_weight) * (t_z/square_size - depth_prior/square_size)。
        depth_weight=0 或未提供 depth_prior 时不生效。
        """
        if self.image is None or self.image_points is None or self.world_points is None:
            raise RuntimeError("Corners/world points not available. Run detect_corners first.")

        img_size = self.image.shape[:2]
        # 使用 x,y 平面点进行线性初值
        world_xy = self.world_points[:, :2]
        r_part, t_part = partial_extrinsics(self.image_points, world_xy, img_size, camera.distortion_center)

        # 构建候选并选择最小 RMS 的解
        candidates = get_full_rotation_matrix(r_part, t_part, self.image_points, img_size, camera.distortion_center)

        def _refine_from_candidate(R_init: np.ndarray, t_init: np.ndarray) -> Tuple[np.ndarray, float]:
            """用 SciPy LM 在 OCam 模型下细化 rvec/tvec。"""
            # 初值：Rodrigues + t，避免 t_z=0 的退化，给一个合理的初值
            rvec0, _ = cv.Rodrigues(R_init)
            tvec0 = t_init.astype(np.float64).copy()
            if abs(tvec0[2]) < 1e-6:
                t_xy = float(np.linalg.norm(tvec0[:2]))
                tvec0[2] = max(1.0, 0.5 * t_xy)

            def _resid(params: np.ndarray) -> np.ndarray:
                rv = params[:3]
                tv = params[3:6]
                Rm, _ = cv.Rodrigues(rv)
                Rt = np.hstack([Rm, tv.reshape(3, 1)])
                proj = camera.world2cam(self.world_points, Rt)
                res = (proj - self.image_points).ravel()
                # 弱深度先验（单位归一化到格子数）
                if depth_prior is not None and depth_weight > 0.0 and self.square_size > 0:
                    tz_norm = tv[2] / float(self.square_size)
                    z0_norm = float(depth_prior) / float(self.square_size)
                    prior_res = np.sqrt(depth_weight) * (tz_norm - z0_norm)
                    res = np.hstack([res, prior_res])
                return res

            x0 = np.hstack([rvec0.ravel(), tvec0.ravel()])
            res = least_squares(_resid, x0, method="lm", max_nfev=200, xtol=1e-10, ftol=1e-10, gtol=1e-10)
            rvec = res.x[:3]
            tvec = res.x[3:6]
            Rm, _ = cv.Rodrigues(rvec)
            Rt_ref = np.hstack([Rm, tvec.reshape(3, 1)])
            err = float(np.linalg.norm(camera.world2cam(self.world_points, Rt_ref) - self.image_points, axis=1).mean())
            return Rt_ref, err

        best_Rt = None
        best_err = np.inf
        for cand in candidates:
            R0 = cand[:, :3]
            t0 = cand[:, 3]
            Rt_ref, err = _refine_from_candidate(R0, t0)
            if err < best_err:
                best_err = err
                best_Rt = Rt_ref

        self.extrinsics_t = best_Rt
        return self.extrinsics_t, float(best_err)

    def visualize(self, camera: Camera, axis_length: float = 65.0) -> np.ndarray:
        if self.image is None or self.extrinsics_t is None:
            raise RuntimeError("Extrinsics not available. Run extract_extrinsic first.")
        overlay = _draw_axes(self.image, camera, self.extrinsics_t, self.square_size, axis_length)
        overlay = _draw_detected_corners(overlay, self.image_points)
        return overlay

    def extract_extrinsic_solvepnp(self, camera: Camera) -> Tuple[np.ndarray, float]:
        """方法2：将像素映射到单位视线，构造针孔归一化坐标，使用 solvePnP 估计外参。

        步骤：
        - image_points -> cam2world 得到单位向量 m=[mx,my,mz]
        - 归一化针孔坐标 uv = (mx/mz, my/mz)，K=I, distCoeffs=None
        - 先 EPnP 粗估，再 ITERATIVE 细化
        - 用 OCam 模型重投影评估 RMS（像素）
        """
        if self.image_points is None or self.world_points is None:
            raise RuntimeError("Corners/world points not available. Run detect_corners first.")

        # 像素 -> 单位视线
        bearings = camera.cam2world(self.image_points.copy())  # (N,3)
        # 归一化针孔坐标
        uv_norm = np.column_stack([bearings[:, 0] / bearings[:, 2],
                                   bearings[:, 1] / bearings[:, 2]]).astype(np.float64)

        Pw = self.world_points.astype(np.float64)
        # OpenCV 期望形状 Nx1x2 或 Nx2，均可
        # 先用 EPNP 初值
        ok, rvec, tvec = cv.solvePnP(Pw, uv_norm, np.eye(3, dtype=np.float64), None,
                                     flags=cv.SOLVEPNP_EPNP)
        if not ok:
            raise RuntimeError("solvePnP (EPNP) failed to find an initial solution")

        # 再用 ITERATIVE 细化
        ok, rvec, tvec = cv.solvePnP(Pw, uv_norm, np.eye(3, dtype=np.float64), None,
                                     rvec, tvec, useExtrinsicGuess=True,
                                     flags=cv.SOLVEPNP_ITERATIVE)
        if not ok:
            raise RuntimeError("solvePnP (ITERATIVE) refinement failed")

        R, _ = cv.Rodrigues(rvec)
        Rt = np.hstack([R, tvec.reshape(3, 1)])

        # 用 OCam 模型评估像素域 RMS
        reproj = camera.world2cam(self.world_points, Rt)
        rms = float(np.linalg.norm(reproj - self.image_points, axis=1).mean())
        return Rt, rms

    def extract_extrinsic_ippe(self,
                               camera: Camera,
                               depth_prior: Optional[float] = None,
                               depth_weight: float = 0.0,
                               mz_threshold: float = 0.0) -> Tuple[np.ndarray, float]:
        """方法3：IPPE 多解 + OCam 像素域打分 + OCam-LM 细化。

        - 将像素角点映射为单位视线，形成针孔归一化坐标；
        - 使用 solvePnPGeneric(IPPE) 输出多个候选位姿；
        - 用 OCam 模型在像素域评分选最优；
        - 在像素域做一次 LM 细化（可选弱深度先验）。
        """
        if self.image_points is None or self.world_points is None:
            raise RuntimeError("Corners/world points not available. Run detect_corners first.")

        # 1) 像素 -> 单位视线 -> 针孔归一化坐标
        bearings = camera.cam2world(self.image_points.copy())  # (N,3)
        if mz_threshold > 0.0:
            mask = bearings[:, 2] > mz_threshold
            if mask.sum() < 6:
                mask = np.ones(len(bearings), dtype=bool)
        else:
            mask = np.ones(len(bearings), dtype=bool)

        uv_norm = np.column_stack([
            bearings[mask, 0] / bearings[mask, 2],
            bearings[mask, 1] / bearings[mask, 2],
        ]).astype(np.float64)
        Pw = self.world_points[mask].astype(np.float64)

        # 2) IPPE 多解
        try:
            ok, rvecs, tvecs, _ = cv.solvePnPGeneric(
                Pw, uv_norm, np.eye(3, dtype=np.float64), None, flags=cv.SOLVEPNP_IPPE
            )
        except TypeError:
            # 某些 OpenCV 版本不返回第四项
            ok, rvecs, tvecs = cv.solvePnPGeneric(
                Pw, uv_norm, np.eye(3, dtype=np.float64), None, flags=cv.SOLVEPNP_IPPE
            )
        if not ok or len(rvecs) == 0:
            raise RuntimeError("solvePnPGeneric(IPPE) failed to produce candidates")

        # 3) 用 OCam 模型在像素域评分
        cands = []
        for rvec, tvec in zip(rvecs, tvecs):
            R, _ = cv.Rodrigues(rvec)
            Rt = np.hstack([R, tvec.reshape(3, 1)])
            reproj = camera.world2cam(self.world_points, Rt)
            rms = float(np.linalg.norm(reproj - self.image_points, axis=1).mean())
            cands.append((Rt, rms))

        Rt0, _ = min(cands, key=lambda x: x[1])

        # 4) OCam-LM 细化（与方法1一致），含可选弱深度先验
        def _resid(params: np.ndarray) -> np.ndarray:
            rv = params[:3]
            tv = params[3:6]
            Rm, _ = cv.Rodrigues(rv)
            Rt = np.hstack([Rm, tv.reshape(3, 1)])
            proj = camera.world2cam(self.world_points, Rt)
            res = (proj - self.image_points).ravel()
            if depth_prior is not None and depth_weight > 0.0 and self.square_size > 0:
                tz_norm = tv[2] / float(self.square_size)
                z0_norm = float(depth_prior) / float(self.square_size)
                # 将先验按点数缩放，便于权重在不同N下保持可比性
                scale = np.sqrt(depth_weight * max(1, 2 * len(self.world_points)))
                prior_res = scale * (tz_norm - z0_norm)
                res = np.hstack([res, prior_res])
            return res

        R0, t0 = Rt0[:, :3], Rt0[:, 3]
        rvec0, _ = cv.Rodrigues(R0)
        tvec0 = t0.astype(np.float64).copy()
        if depth_prior is not None:
            tvec0[2] = float(depth_prior)
        elif abs(tvec0[2]) < 1e-6:
            t_xy = float(np.linalg.norm(tvec0[:2]))
            tvec0[2] = max(1.0, 0.5 * t_xy)

        x0 = np.hstack([rvec0.ravel(), tvec0.ravel()])
        res = least_squares(_resid, x0, method="lm", max_nfev=200, xtol=1e-10, ftol=1e-10, gtol=1e-10)
        rvec = res.x[:3]
        tvec = res.x[3:6]
        Rm, _ = cv.Rodrigues(rvec)
        Rt_ref = np.hstack([Rm, tvec.reshape(3, 1)])
        rms = float(np.linalg.norm(camera.world2cam(self.world_points, Rt_ref) - self.image_points, axis=1).mean())
        return Rt_ref, rms


def _draw_axes(image: np.ndarray,
               camera: Camera,
               extrinsic: np.ndarray,
               square_size: float,
               axis_length: float) -> np.ndarray:
    """Draw the chessboard origin plus X/Y axes on the image."""
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
    cv.putText(overlay, "X", (x_axis[0] + 5, x_axis[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)
    cv.putText(overlay, "Y", (y_axis[0] + 5, y_axis[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
    return overlay


def _draw_detected_corners(image: np.ndarray, corners: np.ndarray) -> np.ndarray:
    overlay = image.copy()
    points = np.round(corners).astype(int)
    for point in points:
        cv.circle(overlay, (int(point[0]), int(point[1])), 3, (255, 0, 0), -1)
    return overlay

"""
python src/pyocamcalib/script/extrinsic_calib2.py ./src/pyocamcalib/checkpoints/calibration/calibration_inhandus_1_10112025_113611.json ./test_images/inhandus_1/fe1_3.jpg 
"""
# 7*7, 57
# 6x4, 200
def main(
    calibration_file: Path = typer.Argument(..., help="Path to the fisheye calibration JSON file."),
    image_path: Path = typer.Argument(..., help="Path to the chessboard image."),
    chessboard_size_row: int = typer.Option(6, help="Number of inner corners along a row."),
    chessboard_size_column: int = typer.Option(4, help="Number of inner corners along a column."),
    square_size: float = typer.Option(200.0, help="Size of a chessboard square (units carry over to translation)."),
    axis_length: float = typer.Option(65.0, help="Axis length expressed in number of squares to draw."),
    output_path: Optional[Path] = typer.Option('./outputs/', help="Optional path to save the overlay image."),
    depth_prior: Optional[float] = typer.Option(1200.0, help="Optional weak prior for tz (same units as square_size)."),
    depth_weight: float = typer.Option(200.0, help="Weak prior weight; 0 disables (suggest 0.1–2.0)."),
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

    # 方法1：基于线性部分 + 候选消歧
    extrinsic_1, rms_1 = my_calib_engine.extract_extrinsic(
        camera,
        depth_prior=depth_prior,
        depth_weight=depth_weight,
    )
    R1 = extrinsic_1[:, :3]
    t1 = extrinsic_1[:, 3]

    # 角度输出
    def _rotation_matrix_to_euler(rotation: np.ndarray) -> np.ndarray:
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

    roll1, pitch1, yaw1 = _rotation_matrix_to_euler(R1)

    typer.echo("Method 1: [R|t] (linear+disambiguation)")
    with np.printoptions(precision=6, suppress=True):
        typer.echo(extrinsic_1)
    typer.echo(f"t1 (units={square_size}): x={t1[0]:.6f}, y={t1[1]:.6f}, z={t1[2]:.6f}")
    typer.echo(f"rpy1 (deg): roll={roll1:.3f}, pitch={pitch1:.3f}, yaw={yaw1:.3f}")
    typer.echo(f"RMS1: {rms_1:.4f} px")

    # 方法3：IPPE 多解 + OCam 像素域打分 + OCam-LM 细化
    extrinsic_3, rms_3 = my_calib_engine.extract_extrinsic_ippe(
        camera,
        depth_prior=depth_prior,
        depth_weight=depth_weight,
        mz_threshold=0.0,
    )
    R3 = extrinsic_3[:, :3]
    t3 = extrinsic_3[:, 3]
    roll3, pitch3, yaw3 = _rotation_matrix_to_euler(R3)

    typer.echo("\nMethod 3: [R|t] (IPPE + OCam refine)")
    with np.printoptions(precision=6, suppress=True):
        typer.echo(extrinsic_3)
    typer.echo(f"t3 (units={square_size}): x={t3[0]:.6f}, y={t3[1]:.6f}, z={t3[2]:.6f}")
    typer.echo(f"rpy3 (deg): roll={roll3:.3f}, pitch={pitch3:.3f}, yaw={yaw3:.3f}")
    typer.echo(f"RMS3: {rms_3:.4f} px")

    # 差异指标（方法1 vs 方法3）
    d_t = np.linalg.norm(t1 - t3)
    d_R = R3 @ R1.T
    angle = np.degrees(np.arccos(np.clip((np.trace(d_R) - 1) / 2.0, -1.0, 1.0)))
    typer.echo(f"\nDelta translation norm (M1 vs M3): {d_t:.6f} (units of square_size)")
    typer.echo(f"Delta rotation angle (M1 vs M3): {angle:.6f} deg")

    # 可视化并保存
    overlay1 = _draw_axes(my_calib_engine.image, camera, extrinsic_1, square_size, axis_length)
    overlay1 = _draw_detected_corners(overlay1, my_calib_engine.image_points)
    overlay3 = _draw_axes(my_calib_engine.image, camera, extrinsic_3, square_size, axis_length)
    overlay3 = _draw_detected_corners(overlay3, my_calib_engine.image_points)

    output_file_path = Path(output_path) / f"{image_path.stem}_axes_m1.jpg"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_file_path), overlay1)
    typer.echo(f"Overlay M1 saved to: {output_file_path}")

    output_file_path3 = Path(output_path) / f"{image_path.stem}_axes_m3.jpg"
    cv.imwrite(str(output_file_path3), overlay3)
    typer.echo(f"Overlay M3 saved to: {output_file_path3}")


if __name__ == "__main__":
    typer.run(main)
