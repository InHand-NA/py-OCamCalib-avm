"""
功能：四相机（front/right/back/left）联合外参标定（AVM场景）。

设计要点：
- 世界坐标系与 front 相机看到的棋盘格坐标系完全一致（即 front 棋盘格为世界坐标原点/朝向）。
- 其余 3 个棋盘（right/back/left）在世界坐标系下的平移由程序预定义（可通过参数覆盖），
  姿态由程序按顺时针每次绕 Z 轴旋转 90° 设定：front=0°，right=-90°，back=-180°，left=-270°。
- 对每个相机，先在各自图像中检测棋盘角点，估计“棋盘坐标系 -> 相机坐标系”的外参 [R|t]，
  再结合“棋盘坐标系 -> 世界坐标系”固定变换，转换为“世界坐标系 -> 相机坐标系”的外参 [R|t]。

实现说明：
- 复用 extrinsic_calib2.py 中的 ExtCalibrationEngine（OCam 模型下的外参估计）。
- 提供可选的调试功能（--debug/--show）以及可视化叠加图保存。
- 详细中文注释，便于理解和二次开发。
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional, Tuple
import json
import time

import numpy as np
import cv2 as cv
import typer
from loguru import logger

from pyocamcalib.modelling.camera import Camera
from pyocamcalib.script.extrinsic_calib2 import ExtCalibrationEngine


app = typer.Typer(help="四相机联合外参标定（AVM）")


# ----------------------------
# 数学/坐标变换辅助
# ----------------------------
def rotz(deg: float) -> np.ndarray:
    """绕 Z 轴旋转的旋转矩阵（右手系，deg 为角度，正方向为逆时针）。"""
    rad = np.deg2rad(deg)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    return np.array([[c, -s, 0.0],
                     [s,  c, 0.0],
                     [0.0, 0.0, 1.0]], dtype=np.float64)


def rpy_from_R(R: np.ndarray) -> Tuple[float, float, float]:
    """将旋转矩阵转换为欧拉角（roll/pitch/yaw, 单位°）。"""
    sy = float(np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    singular = sy < 1e-9
    if not singular:
        roll = np.arctan2(R[2, 1], R[2, 2])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = np.arctan2(R[1, 0], R[0, 0])
    else:
        roll = np.arctan2(-R[1, 2], R[1, 1])
        pitch = np.arctan2(-R[2, 0], sy)
        yaw = 0.0
    return tuple(np.degrees([roll, pitch, yaw]).tolist())


def compose_world_extrinsic(R_c_b: np.ndarray,
                            t_c_b: np.ndarray,
                            R_w_b: np.ndarray,
                            t_w_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将“棋盘->相机”的外参转换为“世界->相机”的外参。

    记：x_c = R_c_b x_b + t_c_b；x_w = R_w_b x_b + t_w_b。
    则：x_c = R_c_w x_w + t_c_w。
    推得：R_c_w = R_c_b R_b_w，其中 R_b_w = R_w_b^T；
          t_c_w = t_c_b - R_c_w t_w_b。
    """
    R_b_w = R_w_b.T
    R_c_w = R_c_b @ R_b_w
    t_c_w = t_c_b - (R_c_w @ t_w_b)
    return R_c_w, t_c_w


def visualize(image: np.ndarray,
              camera: Camera,
              Rt_world_to_cam: np.ndarray,
              square_size: float,
              axis_length: float,
              corners: Optional[np.ndarray] = None,
              title: Optional[str] = None) -> np.ndarray:
    """
    在图像上叠加“世界坐标系”的原点与三轴。

    输入：
    - image: 原始图像（BGR）
    - camera: OCam 模型相机
    - Rt_world_to_cam: 3x4 外参，[R|t]（世界 -> 相机）
    - square_size: 棋盘格边长（与 t 的单位一致）
    - axis_length: 坐标轴长度（以“格”为单位），最终长度 = square_size * axis_length
    - corners: 可选，绘制已检测到的角点
    - title: 可选，左上角标注文本

    返回：叠加后的图像。
    """
    overlay = image.copy()
    axis_extent = float(square_size) * float(axis_length)
    # 世界坐标系下的 4 个点：原点 + X/Y/Z 轴端点
    world_axes = np.array([
        [0.0, 0.0, 0.0],
        [axis_extent, 0.0, 0.0],
        [0.0, axis_extent, 0.0],
        [0.0, 0.0, axis_extent],
    ], dtype=np.float64)
    proj = np.round(camera.world2cam(world_axes, Rt_world_to_cam)).astype(int)
    o = tuple(proj[0])
    x_end = tuple(proj[1])
    y_end = tuple(proj[2])
    z_end = tuple(proj[3])
    # 原点与三轴
    cv.circle(overlay, o, 6, (255, 0, 255), -1)  # magenta origin
    cv.line(overlay, o, x_end, (0, 0, 255), 2)
    cv.line(overlay, o, y_end, (0, 255, 0), 2)
    cv.line(overlay, o, z_end, (255, 0, 0), 2)
    cv.putText(overlay, "Xw", (x_end[0] + 5, x_end[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv.LINE_AA)
    cv.putText(overlay, "Yw", (y_end[0] + 5, y_end[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv.LINE_AA)
    cv.putText(overlay, "Zw", (z_end[0] + 5, z_end[1] + 5), cv.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2, cv.LINE_AA)

    # 角点可视化（可选）
    if corners is not None and len(corners) > 0:
        pts = np.round(corners).astype(int)
        for p in pts:
            cv.circle(overlay, (int(p[0]), int(p[1])), 3, (255, 200, 0), -1)

    if title:
        cv.putText(overlay, title, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2, cv.LINE_AA)

    return overlay


def _pick_and_print_world_points(image: np.ndarray,
                                 camera: Camera,
                                 Rt_world_to_cam: np.ndarray,
                                 count: int = 8,
                                 win_name: str = 'pick') -> None:
    """
    交互拾取像素点，并计算其在世界坐标系 Z=0 平面上的交点坐标，打印输出。

    算法：
    - 已知世界->相机外参 [R|t]；
    - 将像素 (u,v) 通过 OCam 模型 cam.cam2world() 转为相机坐标系单位视线 v_c；
    - 旋转到世界系：v_w = v_c @ R（等价于 v_w = R^T v_c 的行向量形式实现）；
    - 相机中心 C_w = -R^T t；与视线的参数方程 X_w = C_w + λ v_w；
    - 与 Z=0 平面求交：λ = -C_w.z / v_w.z。
    """
    picked: list = []

    def _on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            picked.append([x, y])
            cv.drawMarker(param, (x, y), (0, 255, 255), markerType=cv.MARKER_CROSS, markerSize=12, thickness=2)
            cv.imshow(win_name, param)

    viz = image.copy()
    cv.namedWindow(win_name, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)
    cv.imshow(win_name, viz)
    cv.setMouseCallback(win_name, _on_mouse, viz)

    while len(picked) < int(count):
        if cv.waitKey(10) & 0xFF == 27:  # ESC 退出
            break
    cv.destroyWindow(win_name)

    if not picked:
        logger.warning("未拾取任何像素点。")
        return

    uv = np.asarray(picked, dtype=np.float64)
    R = Rt_world_to_cam[:, :3].astype(np.float64)
    t = Rt_world_to_cam[:, 3].astype(np.float64)
    R_T = R.T
    Cw = -R_T @ t  # 相机中心在世界坐标

    # 像素 -> 相机视线；再旋转到世界系（行向量实现 v_w = v_c @ R）
    rays_cam = camera.cam2world(uv.copy())  # Nx3 单位向量
    rays_w = rays_cam @ R  # 旋转到世界系

    vz = rays_w[:, 2]
    with np.errstate(divide='ignore', invalid='ignore'):
        lamb = -Cw[2] / vz
    invalid = np.abs(vz) < 1e-12
    lamb[invalid] = np.nan

    Xw = Cw[None, :] + lamb[:, None] * rays_w  # Nx3

    for i, (px, pw) in enumerate(zip(uv, Xw)):
        typer.echo(f"[{i}] pixel=({px[0]:.2f}, {px[1]:.2f}) -> world=(X={pw[0]:.6f}, Y={pw[1]:.6f}, Z={pw[2]:.6f})")


def _resolve_calibration_files(calib_dir: Optional[Path],
                               calib_front: Optional[Path],
                               calib_right: Optional[Path],
                               calib_back: Optional[Path],
                               calib_left: Optional[Path]) -> Dict[str, Path]:
    """解析/匹配四路相机内参文件路径。

    约定：目录模式下优先寻找固定命名文件：intrinsic_front.json、intrinsic_right.json、
    intrinsic_back.json、intrinsic_left.json；若不存在，再回退到包含关键字 front/right/back/left 的任意 JSON。
    若通过参数分别指定四个 JSON，则直接使用指定路径。
    """
    def find_in_dir(d: Path, key: str) -> Optional[Path]:
        # 1) 优先固定命名：intrinsic_<key>.json（不区分大小写）
        exact = d / f"intrinsic_{key}.json"
        if exact.is_file():
            return exact
        # 2) 回退：名称包含关键字的 JSON（不区分大小写）
        candidates = sorted([p for p in d.glob("*.json") if key in p.name.lower()])
        return candidates[0] if candidates else None

    out = {}
    if calib_front and calib_right and calib_back and calib_left:
        out = {
            "front": calib_front,
            "right": calib_right,
            "back": calib_back,
            "left": calib_left,
        }
    elif calib_dir is not None:
        d = calib_dir
        out = {
            "front": find_in_dir(d, "front"),
            "right": find_in_dir(d, "right"),
            "back": find_in_dir(d, "back"),
            "left": find_in_dir(d, "left"),
        }
        missing = [k for k, v in out.items() if v is None]
        if missing:
            raise typer.BadParameter(
                f"在目录 {calib_dir} 下未找到如下相机的 JSON: {missing}；"
                f"请按命名约定提供 intrinsic_<front|right|back|left>.json 或使用各自 --calib-<pos> 指定"
            )
    else:
        raise typer.BadParameter("需指定 --calib-dir 或分别指定四个 --calib-<pos> JSON 内参文件")

    for k, p in out.items():
        if not Path(p).is_file():
            raise typer.BadParameter(f"内参文件不存在: {k} -> {p}")
    return out


def _default_board_to_world(
    right_xy: Optional[Tuple[float, float]] = None,
    back_xy: Optional[Tuple[float, float]] = None,
    left_xy: Optional[Tuple[float, float]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    生成四块棋盘相对于“世界坐标系(front 棋盘)”的刚体变换（R_w_b, t_w_b）。

    - 姿态（R_w_b）：固定为 front=I，right 绕 Z 轴 -90°，back -180°，left -270°（顺时针）。
    - 平移（t_w_b）：
        - 若提供测量坐标 right_xy/back_xy/left_xy，则分别使用 [x, y, 0].

    参数单位：所有长度（square_size、right_xy 等）保持一致，例如毫米。
    """

    def _t_xy(xy_opt: Optional[Tuple[float, float]]) -> np.ndarray:
        if xy_opt is not None:
            x, y = float(xy_opt[0]), float(xy_opt[1])
            return np.array([x, y, 0.0], dtype=np.float64)

    return {
        "front": (np.eye(3, dtype=np.float64), np.array([0.0, 0.0, 0.0], dtype=np.float64)),
        "right": (rotz(-90.0), _t_xy(right_xy)),
        "back":  (rotz(-180.0), _t_xy(back_xy)),
        "left":  (rotz(-270.0), _t_xy(left_xy)),
    }


@app.command()
def main(
    # 内参：支持目录匹配或分别指定
    calib_dir: Optional[Path] = typer.Option(None, help="包含四路相机内参 JSON 的目录（按 front/right/back/left 关键字匹配）"),
    calib_front: Optional[Path] = typer.Option(None, help="front 相机内参 JSON 路径"),
    calib_right: Optional[Path] = typer.Option(None, help="right 相机内参 JSON 路径"),
    calib_back: Optional[Path] = typer.Option(None, help="back 相机内参 JSON 路径"),
    calib_left: Optional[Path] = typer.Option(None, help="left 相机内参 JSON 路径"),

    # 图像：四张单次采集的棋盘图
    img_front: Path = typer.Argument(..., help="front 相机棋盘图像"),
    img_right: Path = typer.Argument(..., help="right 相机棋盘图像"),
    img_back: Path = typer.Argument(..., help="back 相机棋盘图像"),
    img_left: Path = typer.Argument(..., help="left 相机棋盘图像"),

    # 棋盘参数
    chessboard_size_row: int = typer.Option(6, help="棋盘内角点沿行方向个数"),
    chessboard_size_column: int = typer.Option(4, help="棋盘内角点沿列方向个数"),
    square_size: float = typer.Option(200.0, help="棋盘单元大小（与 t 的单位一致）"),
    gap_squares: float = typer.Option(None, help="棋盘之间中心间隔（单位：格），默认取 chessboard_size_row"),

    # 可视化与调试
    axis_length: float = typer.Option(20.0, help="坐标轴长度（单位：格）用于叠加图"),
    output_dir: Path = typer.Option(Path("./outputs"), help="输出叠加图目录"),
    checkpoints_dir: Path = typer.Option(Path("src/pyocamcalib/checkpoints/calibration"), help="保存联合外参 JSON 的目录"),
    depth_prior: Optional[float] = typer.Option(1200.0, help="弱深度先验 tz（与 square_size 同单位），None 表示关闭"),
    depth_weight: float = typer.Option(200.0, help="弱先验权重，0 表示不使用"),
    # 测量得到的 right/back/left 棋盘原点在 front 参考坐标中的 (x, y)
    right_xy: Optional[Tuple[float, float]] = typer.Option(None, help="right 棋盘原点在 front-世界坐标中的 (x, y)"),
    back_xy: Optional[Tuple[float, float]] = typer.Option(None, help="back 棋盘原点在 front-世界坐标中的 (x, y)"),
    left_xy: Optional[Tuple[float, float]] = typer.Option(None, help="left 棋盘原点在 front-世界坐标中的 (x, y)"),
    show: bool = typer.Option(False, help="是否弹窗显示可视化结果（调试）"),
    debug: bool = typer.Option(False, help="打印详细调试信息"),
    verify: bool = typer.Option(True, help="联合标定后交互拾取每路 8 个像素点并输出世界坐标"),
):
    """四路相机联合外参标定入口。"""
    if gap_squares is None:
        gap_squares = float(chessboard_size_row)

    # 解析内参文件
    calib_map = _resolve_calibration_files(calib_dir, calib_front, calib_right, calib_back, calib_left)
    cam_map: Dict[str, Camera] = {k: Camera.load_parameters_json(str(v)) for k, v in calib_map.items()}

    # 统一参数
    chessboard_size = (int(chessboard_size_row), int(chessboard_size_column))
    image_map: Dict[str, Path] = {
        "front": img_front,
        "right": img_right,
        "back": img_back,
        "left": img_left,
    }

    # 预定义“棋盘->世界”刚体变换（R_w_b, t_w_b）
    Twb = _default_board_to_world(right_xy=right_xy, back_xy=back_xy, left_xy=left_xy)

    # 逐相机：检测角点 -> 估计 (棋盘->相机) 外参 -> 转到 (世界->相机)
    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for key in ("front", "right", "back", "left"):
        img_path = image_map[key]
        if not Path(img_path).is_file():
            raise typer.BadParameter(f"图像不存在: {key} -> {img_path}")

        camera = cam_map[key]
        # 使用图像所在目录作为 working_dir，避免 ExtCalibrationEngine 初始化期望目录下无图片
        engine = ExtCalibrationEngine(str(Path(img_path).parent), chessboard_size, key, square_size)
        ok = engine.detect_corners2(img_path, check=False)
        if not ok:
            raise RuntimeError(f"{key} 未检测到有效棋盘角点: {img_path}")

        Rt_cb, rms_px = engine.extract_extrinsic(camera, depth_prior=depth_prior, depth_weight=depth_weight)
        R_cb = Rt_cb[:, :3]
        t_cb = Rt_cb[:, 3]

        # 可视化1（棋盘坐标轴，便于核验局部外参）：
        overlay_board = engine.visualize(camera, axis_length=axis_length)
        out_img = output_dir / f"{Path(img_path).stem}_axes_board_{key}.jpg"
        cv.imwrite(str(out_img), overlay_board)

        # 转换到世界坐标系
        R_w_b, t_w_b = Twb[key]
        R_c_w, t_c_w = compose_world_extrinsic(R_cb, t_cb, R_w_b, t_w_b)
        roll, pitch, yaw = rpy_from_R(R_c_w)

        results[key] = {
            "rms_px": float(rms_px),
            "Rt_board_to_cam": Rt_cb.tolist(),
            "Rt_world_to_cam": np.hstack([R_c_w, t_c_w.reshape(3, 1)]).tolist(),
            "rpy_world_to_cam_deg": [float(roll), float(pitch), float(yaw)],
        }

        if debug:
            logger.info(f"[{key}] RMS={rms_px:.4f} px")
            logger.info(f"[{key}] R_c_b=\n{R_cb}")
            logger.info(f"[{key}] t_c_b={t_cb}")
            logger.info(f"[{key}] R_w_b=\n{R_w_b}")
            logger.info(f"[{key}] t_w_b={t_w_b}")
            logger.info(f"[{key}] R_c_w=\n{R_c_w}")
            logger.info(f"[{key}] t_c_w={t_c_w}")

        # 可视化2（世界坐标轴，便于多相机关联核验）：
        Rt_c_w = np.hstack([R_c_w, t_c_w.reshape(3, 1)])
        overlay_world = visualize(engine.image, camera, Rt_c_w, square_size, axis_length, corners=engine.image_points,
                                  title=f"{key}: world axes")
        out_img_w = output_dir / f"{Path(img_path).stem}_axes_world_{key}.jpg"
        cv.imwrite(str(out_img_w), overlay_world)

        if show:
            cv.imshow(f"{key}-board", overlay_board)
            cv.imshow(f"{key}-world", overlay_world)

        # 交互验证：拾取 8 个像素点，打印世界坐标（Z=0 平面交点）
        if verify:
            _pick_and_print_world_points(engine.image, camera, Rt_c_w, count=8, win_name=f"pick-{key}")

    # 弹窗展示
    if show:
        logger.info("按任意键关闭所有窗口...")
        cv.waitKey(0)
        for key in ("front", "right", "back", "left"):
            try:
                cv.destroyWindow(f"{key}-board")
                cv.destroyWindow(f"{key}-world")
            except Exception:
                pass

    # 保存联合外参 JSON
    ts = time.strftime("%Y%m%d_%H%M%S")
    extr_path = checkpoints_dir / f"avm_extrinsics_{ts}.json"
    with open(extr_path, "w", encoding="utf-8") as f:
        json.dump({
            "square_size": float(square_size),
            "chessboard_size": [int(chessboard_size_row), int(chessboard_size_column)],
            "gap_squares": float(gap_squares),
            "board2world": {
                k: {"R_w_b": Twb[k][0].tolist(), "t_w_b": Twb[k][1].tolist()} for k in ("front", "right", "back", "left")
            },
            "cameras": results,
        }, f, ensure_ascii=False, indent=2)
    typer.echo(f"联合外参已保存: {extr_path}")


if __name__ == "__main__":
    app()
