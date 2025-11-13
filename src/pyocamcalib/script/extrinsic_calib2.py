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
from pyocamcalib.modelling.utils import get_files, check_detection


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
        self.chessboard_size = chessboard_size
        self.square_size = square_size
        self.image_points = None
        self.world_points = None
        self.image = None
        self.image_path = None
        self.extrinsics_t = None
        self.cam_name = camera_name

    def my_generate_world_points(self):
        cols, rows = self.chessboard_size
        # Object points in board frame (single template reused per image)
        objp = np.zeros((1, cols * rows, 3), np.float32)
        objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
        objp *= float(self.square_size)
        return objp

    def detect_corners2(self, image_file_path: Path, check: bool = False, max_height=520) -> bool:
        """Detect chessboard corners using classic OpenCV routine.

        Reference: see opencv_fe_intrinsicCalib2.py corner detection (findChessboardCorners + subpix).

        - Uses flags: ADAPTIVE_THRESH | FAST_CHECK | NORMALIZE_IMAGE
        - Refines with cornerSubPix
        - Populates self.image_points (Nx2), self.world_points (Nx3), self.image, self.image_path

        :param image_file_path: Path to a single chessboard image
        :param check: If True, launch interactive check/edit UI
        :return: True if detection succeeded, False otherwise
        """
        img = cv.imread(str(image_file_path))
        if img is None or img.size == 0:
            logger.error(f"Unable to read image: {image_file_path}")
            return False

        self.image_path = str(image_file_path)
        self.image = img

        gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
        det_flags = (
            cv.CALIB_CB_ADAPTIVE_THRESH
            | cv.CALIB_CB_FAST_CHECK
            | cv.CALIB_CB_NORMALIZE_IMAGE
        )
        found, corners = cv.findChessboardCorners(gray, self.chessboard_size, det_flags)
        if not found or corners is None:
            logger.warning(f"Chessboard not found: {image_file_path}")
            return False

        # Subpixel refinement
        subpix_criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 30, 0.1)
        cv.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)

        draw_img = self.image.copy()
        cv.drawChessboardCorners(self.image, self.chessboard_size, corners, True)

        # Generate corresponding world points (Z=0 plane with unit set by square_size)
        #world_points = generate_checkerboard_points(self.chessboard_size, self.square_size, z_axis=True)
        world_points = self.my_generate_world_points()

        # Store detections (keep ordering consistent with existing detect_corners)
        corners2d = np.squeeze(corners).astype(np.float64)
        self.image_points = corners2d #corners2d[::-1]

        self.world_points = np.squeeze(world_points)

        if check:
            try:
                check_detection(self.image_points.copy(), img)
            except Exception:
                pass

        logger.info("Chessboard corners detected (detect_corners2)")
        return True

    def visualize(self, camera: Camera, axis_length: float = 3.0) -> np.ndarray:
        """在图像上叠加棋盘坐标系原点与 X/Y/Z 三轴，并绘制检测到的角点。

        - 使用 self.extrinsics_t 作为 [R|t]（棋盘->相机）。
        - 坐标轴长度 = self.square_size * axis_length。
        - 若未先提取外参或未检测到角点，将抛出异常提示。
        """
        if self.image is None or self.extrinsics_t is None:
            raise RuntimeError("Extrinsics not available. Run extract_extrinsic first.")

        overlay = _draw_axes(self.image, camera, self.extrinsics_t, self.square_size, axis_length)

        # 绘制角点（浅橙色小圆点）
        if self.image_points is not None:
            pts = np.round(self.image_points).astype(int)
            for p in pts:
                cv.circle(overlay, (int(p[0]), int(p[1])), 3, (255, 200, 0), -1)

        return overlay

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

    def save_extrinsic_txt(self, output_path: Optional[Path] = None) -> Path:
        """Export the current single-view extrinsic [R|t] to a .txt file.

        Format mirrors modelling.calibration.CalibrationEngine.save_extrinsic_txt:
        - translation (same unit as square_size; e.g., mm)
        - rotation as roll/pitch/yaw (degrees)

        If output_path is None, writes to
        ./src/pyocamcalib/checkpoints/calibration/ocamcalib_extrinsic_<cam_name>.txt
        """
        if self.extrinsics_t is None:
            raise ValueError("Extrinsic parameters are empty. Run extract_extrinsic() first.")

        Rt = np.asarray(self.extrinsics_t, dtype=np.float64)
        R = Rt[:, :3]
        t = Rt[:, 3]

        def rotation_matrix_to_euler(rot: np.ndarray) -> Tuple[float, float, float]:
            sy = float(np.sqrt(rot[0, 0] ** 2 + rot[1, 0] ** 2))
            singular = sy < 1e-9
            if not singular:
                roll = np.arctan2(rot[2, 1], rot[2, 2])
                pitch = np.arctan2(-rot[2, 0], sy)
                yaw = np.arctan2(rot[1, 0], rot[0, 0])
            else:
                roll = np.arctan2(-rot[1, 2], rot[1, 1])
                pitch = np.arctan2(-rot[2, 0], sy)
                yaw = 0.0
            ang = np.degrees([roll, pitch, yaw])
            return float(ang[0]), float(ang[1]), float(ang[2])

        roll, pitch, yaw = rotation_matrix_to_euler(R)

        if output_path is None:
            output_path = Path(f'./src/pyocamcalib/checkpoints/calibration/ocamcalib_extrinsic_{self.cam_name}.txt')
        output_path.parent.mkdir(parents=True, exist_ok=True)

        img_tag = self.image_path if self.image_path is not None else "<image>"
        lines = []
        lines.append(f"{img_tag}:\n")
        lines.append("translation (units)\n")
        lines.append(f"  {t[0]:.9g}        # trans_x\n")
        lines.append(f"  {t[1]:.9g}        # trans_y\n")
        lines.append(f"  {t[2]:.9g}        # trans_z\n")
        lines.append("rotation (degree)\n")
        lines.append(f"  {roll:.9g}        # roll\n")
        lines.append(f"  {pitch:.9g}        # pitch\n")
        lines.append(f"  {yaw:.9g}        # yaw\n\n")

        with open(output_path, 'w', encoding='utf-8') as f:
            f.writelines(lines)

        logger.info(f"Extrinsic file exported to {output_path}")
        return output_path

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
        [0.0, 0.0, -axis_extent],
    ])
    projected = np.round(camera.world2cam(axis_points, extrinsic)).astype(int)
    origin = tuple(projected[0])
    x_axis = tuple(projected[1])
    y_axis = tuple(projected[2])
    z_axis = tuple(projected[3])
    cv.circle(overlay, origin, 6, (0, 0, 255), -1)
    cv.line(overlay, origin, x_axis, (0, 0, 255), 2)
    cv.line(overlay, origin, y_axis, (0, 255, 0), 2)
    cv.line(overlay, origin, z_axis, (255, 0, 0), 2)
    cv.putText(overlay, "X", (x_axis[0] + 5, x_axis[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)
    cv.putText(overlay, "Y", (y_axis[0] + 5, y_axis[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
    cv.putText(overlay, "Z", (z_axis[0] + 5, z_axis[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv.LINE_AA)
    return overlay


def eval_extrinsic():
    """Interactive check: pick 4 pixels and print their world coords on Z=0.

    This function relies on globals set by main():
      - _LAST_CAMERA: Camera
      - _LAST_IMAGE: np.ndarray
      - _LAST_EXTRINSIC: 3x4 [R|t]
    """
    try:
        cam = globals().get("_LAST_CAMERA", None)
        img = globals().get("_LAST_IMAGE", None)
        Rt = globals().get("_LAST_EXTRINSIC", None)
    except Exception:
        cam, img, Rt = None, None, None

    if cam is None or img is None or Rt is None:
        logger.error("eval_extrinsic requires a computed extrinsic and loaded camera/image (run main first).")
        return

    picked = []

    def _on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            picked.append([x, y])
            cv.drawMarker(param, (x, y), (0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=12, thickness=2)
            cv.imshow('pick-8', param)

    viz = img.copy()
    cv.namedWindow('pick-8', cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.imshow('pick-8', viz)
    cv.setMouseCallback('pick-8', _on_mouse, viz)

    while len(picked) < 8:
        if cv.waitKey(10) & 0xFF == 27:  # ESC to quit early
            break
    cv.destroyWindow('pick-8')

    if len(picked) == 0:
        logger.warning("No points picked.")
        return

    uv = np.asarray(picked, dtype=np.float64)

    # Compute world intersection with Z=0 plane
    R = Rt[:, :3].astype(np.float64)
    t = Rt[:, 3].astype(np.float64)
    R_T = R.T
    Cw = -R_T @ t  # camera center in world coordinates

    # rays in camera coords, then rotate to world coords
    rays_cam = cam.cam2world(uv.copy())  # Nx3 unit
    rays_w = rays_cam @ R_T.T  # rotate to world: v_w = R^T * v_c

    vz = rays_w[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        lamb = -Cw[2] / vz
    # handle nearly parallel rays (vz ~ 0): set NaN
    invalid = np.abs(vz) < 1e-12
    lamb[invalid] = np.nan

    Xw = Cw[None, :] + lamb[:, None] * rays_w  # Nx3

    for i, (px, pw) in enumerate(zip(uv, Xw)):
        typer.echo(f"[{i}] pixel=({px[0]:.2f}, {px[1]:.2f}) -> world=(X={pw[0]:.6f}, Y={pw[1]:.6f}, Z={pw[2]:.6f})")


"""
python src/pyocamcalib/script/extrinsic_calib2.py /home/zyb/avm/py-OCamCalib/src/pyocamcalib/checkpoints/calibration/calibration_inhandus_1_12112025_140617.json  /home/zyb/avm/py-OCamCalib/test_images/ext_test/ext_test3.jpg


python src/pyocamcalib/script/extrinsic_calib2.py usb_front src/pyocamcalib/checkpoints/calibration/calibration_usb_front_13112025_102405.json  /home/zyb/avm/py-OCamCalib/test_images/ext_test/usb_front_2.jpg
"""
# 7*7, 57
# 6x4, 200
def main(
    camera_name: str = typer.Argument(..., help='Camera name'),
    calibration_file: Path = typer.Argument(..., help="Path to the fisheye calibration JSON file."),
    image_path: Path = typer.Argument(..., help="Path to the chessboard image."),
    chessboard_size_row: int = typer.Option(6, help="Number of inner corners along a row."),
    chessboard_size_column: int = typer.Option(4, help="Number of inner corners along a column."),
    square_size: float = typer.Option(200.0, help="Size of a chessboard square (units carry over to translation)."),
    axis_length: float = typer.Option(3.0, help="Axis length expressed in number of squares to draw."),
    output_path: Optional[Path] = typer.Option('./outputs/', help="Optional path to save the overlay image."),
    depth_prior: Optional[float] = typer.Option(None, help="Optional weak prior for tz (same units as square_size)."),
    depth_weight: float = typer.Option(0, help="Weak prior weight; 0 disables (suggest 0.1–2.0)."),
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
    image = cv.imread(str(image_path))
    if image is None:
        raise typer.BadParameter(f"Unable to read image: {image_path}")

    camera = Camera.load_parameters_json(str(calibration_file))
    pattern_size = (chessboard_size_row, chessboard_size_column)

    chessboard_size = (chessboard_size_row, chessboard_size_column)
    my_calib_engine = ExtCalibrationEngine(working_dir, chessboard_size, camera_name, square_size)
    my_calib_engine.detect_corners2(image_path, check=True, max_height=520)

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

    # 可视化并保存
    overlay1 = _draw_axes(my_calib_engine.image, camera, extrinsic_1, square_size, axis_length)
    #overlay1 = _draw_detected_corners(overlay1, my_calib_engine.image_points)

    output_file_path = Path(output_path) / f"{image_path.stem}_axes_m1.jpg"
    output_file_path.parent.mkdir(parents=True, exist_ok=True)
    cv.imwrite(str(output_file_path), overlay1)
    typer.echo(f"Overlay M1 saved to: {output_file_path}")

    # 导出外参到 txt（单视图）
    try:
        out_txt = my_calib_engine.save_extrinsic_txt()
        typer.echo(f"Extrinsic saved to: {out_txt}")
    except Exception as e:
        typer.echo(f"Failed to export extrinsic txt: {e}")

    # Cache for interactive eval_extrinsic()
    globals()["_LAST_CAMERA"] = camera
    globals()["_LAST_IMAGE"] = my_calib_engine.image
    globals()["_LAST_EXTRINSIC"] = extrinsic_1
    eval_extrinsic()


if __name__ == "__main__":
    typer.run(main)
