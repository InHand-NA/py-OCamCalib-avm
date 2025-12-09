"""
功能：四相机（front/right/back/left）联合外参标定（AVM场景）。

设计要点：
- 世界坐标系与 front 相机看到的棋盘格坐标系完全一致（即 front 棋盘格为世界坐标原点/朝向）。
- 其余 3 个棋盘（right/back/left）在世界坐标系下的平移由程序预定义，
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

# 代码内预定义：right/back/left 棋盘原点在 front-世界坐标中的 (x, y)
DEFAULT_RIGHT_XY = (317, 315)
DEFAULT_BACK_XY = (150, 785)
DEFAULT_LEFT_XY = (-163, 465)

app = typer.Typer(help="四相机联合外参标定（AVM）")


# ----------------------------
# 数学/坐标变换辅助
# ----------------------------
def rotz(deg: float) -> np.ndarray:
    """绕 Z 轴旋转的旋转矩阵（右手系，deg 为角度，正方向为逆时针）。"""
    rad = np.deg2rad(deg)
    c, s = float(np.cos(rad)), float(np.sin(rad))
    if abs(c) < 1e-12:
        c = 0.0
    if abs(s) < 1e-12:
        s = 0.0
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


def compose_world_extrinsic(R_b_c: np.ndarray,
                            t_b_c: np.ndarray,
                            R_w_b: np.ndarray,
                            t_w_b: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    将“世界->棋盘”,“棋盘->相机”的外参转换为“世界->相机”的外参。
    """
    R_b_c = np.asarray(R_b_c, dtype=np.float64)
    t_b_c = np.asarray(t_b_c, dtype=np.float64).reshape(3)
    R_w_b = np.asarray(R_w_b, dtype=np.float64)
    t_w_b = np.asarray(t_w_b, dtype=np.float64).reshape(3)

    # 组合变换：X_c = R_b_c (R_w_b X_w + t_w_b) + t_b_c
    R_w_c = R_b_c @ R_w_b
    t_w_c = R_b_c @ t_w_b + t_b_c
    return R_w_c, t_w_c


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
        [0.0, 0.0, -axis_extent],
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
                                 count: int = 4,
                                 win_name: str = 'pick') -> None:
    """
    交互拾取像素点，并计算其在世界坐标系 Z=0 平面上的交点坐标，打印输出。
    """
    if image is None or image.size == 0:
        return

    Rt = np.asarray(Rt_world_to_cam, dtype=np.float64)
    if Rt.shape != (3, 4):
        raise ValueError("Rt_world_to_cam must be 3x4 [R|t] (world->camera)")

    R_wc = Rt[:, :3]
    t_wc = Rt[:, 3]
    R_cw = R_wc.T
    C_w = -R_cw @ t_wc

    img_disp = image.copy()
    clicks = []

    def on_mouse(event, x, y, flags, param):
        if event != cv.EVENT_LBUTTONDOWN:
            return
        if len(clicks) >= count:
            return

        uv = np.array([[float(x), float(y)]], dtype=np.float64)
        v_c = camera.cam2world(uv)[0]  # unit ray in camera frame
        d_w = R_cw @ v_c  # ray direction in world frame

        dz = float(d_w[2])
        if abs(dz) < 1e-12:
            print(f"[{win_name}] pixel=({x}, {y}) -> ray parallel to Z=0 plane; skip")
            return

        s = -float(C_w[2]) / dz
        X_w = C_w + s * d_w
        X_w[2] = 0.0  # enforce plane for numerical stability

        idx = len(clicks) + 1
        clicks.append(((x, y), X_w.copy()))

        cv.circle(img_disp, (int(x), int(y)), 4, (0, 255, 255), -1)
        cv.putText(img_disp, str(idx), (int(x) + 6, int(y) - 6), cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2,
                   cv.LINE_AA)
        cv.imshow(win_name, img_disp)

        print(f"[{win_name}] #{idx} pixel=({x:.1f},{y:.1f}) -> world(X,Y,Z=0)=({X_w[0]:.6f},{X_w[1]:.6f},0.0)")

    cv.namedWindow(win_name, cv.WINDOW_NORMAL)
    cv.imshow(win_name, img_disp)
    cv.setMouseCallback(win_name, on_mouse)

    while len(clicks) < int(count):
        if cv.waitKey(10) & 0xFF in (27, ord('q')):  # ESC or 'q' to abort early
            break

    try:
        cv.destroyWindow(win_name)
    except Exception:
        pass


def _resolve_calibration_files(calib_dir: Optional[Path]) -> Dict[str, Path]:
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
    if calib_dir is not None:
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
        raise typer.BadParameter("需指定 --calib-dir")

    for k, p in out.items():
        if not Path(p).is_file():
            raise typer.BadParameter(f"内参文件不存在: {k} -> {p}")
    return out


def _default_world2boards(
    right_xy: Optional[Tuple[float, float]] = None,
    back_xy: Optional[Tuple[float, float]] = None,
    left_xy: Optional[Tuple[float, float]] = None,
) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """
    生成世界坐标系到当前棋盘坐标系的变换矩阵。所有坐标系都采用右手坐标系。
    - front棋盘坐标系： 与世界坐标系姿态完全重合。
    - right棋盘坐标系： 原点在世界坐标系的位置为right_xy, z=0. 方向绕Z轴旋转90度。
    - back棋盘坐标系： 原点在世界坐标系的位置为back_xy, z=0. 方向绕Z轴旋转180度。
    - left棋盘坐标系： 原点在世界坐标系的位置为left_xy, z=0. 方向绕Z轴旋转270度。

    返回值实例:
    result = {
    "front": [R, t],
    "right": [R, t],
    "back": [R, t],
    "left": [R, t],
    }
    """
    rx, ry = right_xy if right_xy is not None else DEFAULT_RIGHT_XY
    bx, by = back_xy if back_xy is not None else DEFAULT_BACK_XY
    lx, ly = left_xy if left_xy is not None else DEFAULT_LEFT_XY

    # 世界->棋盘 的旋转：front=0°, right=-90°, back=-180°, left=-270°（顺时针棋盘相对世界为负角，逆变换取正角）
    R_w_f = np.eye(3, dtype=np.float64)
    R_w_r = rotz(-90.0)
    R_w_ba = rotz(-180.0)
    R_w_l = rotz(-270.0)

    # 平移：若棋盘原点在世界坐标为 p_w，则 x_b = R x_w + t，需满足 x_b=0 当 x_w=p_w -> t = -R p_w
    p_w_f = np.array([0.0, 0.0, 0.0], dtype=np.float64)
    p_w_r = np.array([float(rx), float(ry), 0.0], dtype=np.float64)
    p_w_b = np.array([float(bx), float(by), 0.0], dtype=np.float64)
    p_w_l = np.array([float(lx), float(ly), 0.0], dtype=np.float64)

    t_w_f = -R_w_f @ p_w_f
    t_w_r = -R_w_r @ p_w_r
    t_w_b = -R_w_ba @ p_w_b
    t_w_l = -R_w_l @ p_w_l

    return {
        "front": (R_w_f, t_w_f),
        "right": (R_w_r, t_w_r),
        "back": (R_w_ba, t_w_b),
        "left": (R_w_l, t_w_l),
    }


def _resolve_image_files(images_dir: Path) -> Dict[str, Path]:
    """从目录中解析 front/right/back/left 的棋盘图像路径（.jpg）。

    规则：文件名以 front/right/back/left 开头，后缀为 .jpg（不区分大小写）。
    同一前缀若匹配到多张，取字典序第一张；若缺失则报错。
    """
    if not images_dir.is_dir():
        raise typer.BadParameter(f"图像目录不存在: {images_dir}")
    keys = ["front", "right", "back", "left"]
    out: Dict[str, Path] = {}
    files = sorted(list(images_dir.glob("*.jpg")) + list(images_dir.glob("*.JPG")))
    name_map = {}
    for p in files:
        name_map.setdefault(p.name.lower(), p)
    for k in keys:
        candidates = [p for p in files if p.name.lower().startswith(k)]
        if not candidates:
            raise typer.BadParameter(f"目录 {images_dir} 下未找到以 '{k}' 开头且以 .jpg 结尾的图像")
        out[k] = sorted(candidates)[0]
    return out




def get_camera_center(camera_params) -> Tuple[float, float, float]:
    """根据外参标定结果，计算 4 个相机在世界坐标系下的平面中心点。

    期望输入格式与本脚本导出的联合外参 JSON 一致：

    - ``camera_params`` 可以是完整 JSON 字典，包含 ``"cameras"`` 键；
      也可以直接是 ``{front/right/back/left: {...}}`` 的相机字典。
    - 每个相机条目应包含 ``"cam2world"`` 子字典，且其中 ``"xyz"`` 为长度为 3
      的可迭代对象，对应相机在世界坐标中的平移 ``(x, y, z)``。

    返回值为所有相机中心在世界坐标系 X-Y 平面的算术平均值 ``(center_x, center_y)``。
    """

    # 传入标定的results
    cams = camera_params

    # 收集每个相机在世界坐标系下的 (x, y)
    front_xyz = cams["front"]["cam2world"]["xyz"]
    right_xyz = cams["right"]["cam2world"]["xyz"]
    back_xyz = cams["back"]["cam2world"]["xyz"]
    left_xyz = cams["left"]["cam2world"]["xyz"]

    center_x = (right_xyz[0] + left_xyz[0]) / 2.0
    center_y = (front_xyz[1] + back_xyz[1]) / 2.0
    center_z = 0.0
    return float(center_x), float(center_y), float(center_z)


def get_ego2world(camera_params):
    """计算ego坐标到世界坐标的转换矩阵。
    ego坐标定义：
    - 原点：4个相机的中心，z=0.
    - x轴正方向： world坐标系Y轴负方向;
    - y轴正方向： world坐标系X轴负方向;
    - z轴正方向： world坐标系Z轴负方向;
    """
    camera_center_xyz = get_camera_center(camera_params)

    # ego 坐标轴相对于世界坐标轴的朝向：
    # x_e -> -y_w, y_e -> -x_w, z_e -> -z_w
    R_ew = np.array(
        [
            [0.0, -1.0, 0.0],   # e_x, e_y, e_z 在 world x 分量
            [-1.0, 0.0, 0.0],   # 对应的 world y 分量
            [0.0, 0.0, -1.0],   # 对应的 world z 分量
        ],
        dtype=np.float64,
    )
    t_ew = np.array(camera_center_xyz, dtype=np.float64)

    return R_ew, t_ew



def get_cam2camb():
    """获取cam坐标到camB坐标的转换参数;
    camB坐标系定义：
    - 原点与cam坐标系原点重合;
    - X = -cam.z
    - Y = -cam.x
    - Z = -cam.y
    """
    # cam -> camB: [X_B, Y_B, Z_B]^T = R_cb [X_c, Y_c, Z_c]^T
    # 其中 X_B = -Z_c, Y_B = -X_c, Z_B = -Y_c
    R_ccb = np.array(
        [
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float64,
    )
    t_ccb = np.zeros(3, dtype=np.float64)
    return R_ccb, t_ccb


def save_extrinsic_txt(calib_txt_dir, cam_name, xyz, rpy_deg):
    """ Save extrinsic txt file
    """
    lines = []
    lines.append("#translation (units)\n")
    lines.append(f"  {xyz[0]:.9g}        # trans_x\n")
    lines.append(f"  {xyz[1]:.9g}        # trans_y\n")
    lines.append(f"  {xyz[2]:.9g}        # trans_z\n")

    lines.append("#rotation (degree)\n")
    lines.append(f"  {rpy_deg[0]:.9g}        # roll\n")
    lines.append(f"  {rpy_deg[1]:.9g}        # pitch\n")
    lines.append(f"  {rpy_deg[2]:.9g}        # yaw\n\n")

    if cam_name == 'back':
        extr_path_txt = calib_txt_dir / "extrinsic_camera_rear.txt"
    else:
        extr_path_txt = calib_txt_dir / f"extrinsic_camera_{cam_name}.txt"
    with open(extr_path_txt, 'w', encoding='utf-8') as f_txt:
        f_txt.writelines(lines)
        print(f"Save extrinsic params at {extr_path_txt} for camera {cam_name}")


"""
python src/pyocamcalib/script/associate_extrinsic_calib.py ./test_images/usb_cameras_003
"""
@app.command()
def main(
    # 内参：支持目录匹配或分别指定
    calib_dir: Optional[Path] = typer.Option("src/pyocamcalib/checkpoints/usb_cameras", help="包含四路相机内参 JSON 的目录（按 front/right/back/left 关键字匹配）"),
    # 图像：提供包含四张图像的目录（文件名以 front/right/back/left 开头，并以 .jpg 结尾）
    images_dir: Path = typer.Argument(..., help="包含四张图像的目录（front/right/back/left*.jpg）"),

    # 棋盘参数
    chessboard_size_column: int = typer.Option(6, help="棋盘内角点沿X方向个数"),
    chessboard_size_row: int = typer.Option(4, help="棋盘内角点沿Y方向个数"),
    square_size: float = typer.Option(30.0, help="棋盘单元大小（与 t 的单位一致）"),

    # 可视化与调试
    axis_length: float = typer.Option(3.0, help="坐标轴长度（单位：格）用于叠加图"),
    output_dir: Path = typer.Option(Path("./outputs"), help="输出叠加图目录"),
    checkpoints_dir: Path = typer.Option(Path("outputs/assosicate"), help="保存联合外参 JSON 的目录"),
    depth_prior: Optional[float] = typer.Option(None, help="弱深度先验 tz（与 square_size 同单位），None 表示关闭"),
    depth_weight: float = typer.Option(0.0, help="弱先验权重，0 表示不使用"),
    
    show: bool = typer.Option(False, help="是否弹窗显示可视化结果（调试）"),
    debug: bool = typer.Option(False, help="打印详细调试信息"),
    verify: bool = typer.Option(False, help="联合标定后交互拾取每路 8 个像素点并输出世界坐标"),
):
    """四路相机联合外参标定入口。"""
    # 固定使用“行内角点个数 * square_size”作为相邻棋盘中心的间隔，不再从命令行传入。

    # 解析内参文件
    calib_map = _resolve_calibration_files(calib_dir)
    cam_map: Dict[str, Camera] = {k: Camera.load_parameters_json(str(v)) for k, v in calib_map.items()}

    # 统一参数
    chessboard_size = (int(chessboard_size_column), int(chessboard_size_row))
    image_map: Dict[str, Path] = _resolve_image_files(images_dir)

    # 将联合标定结果输出到子目录：<checkpoints_dir>/<images_dir.name>
    checkpoints_dir = checkpoints_dir / images_dir.name

    # 预定义“世界->棋盘”刚体变换（R_wb, t_wb）
    Twb = _default_world2boards(
        right_xy=DEFAULT_RIGHT_XY,
        back_xy=DEFAULT_BACK_XY,
        left_xy=DEFAULT_LEFT_XY
    )

    # 逐相机：检测角点 -> 估计 (棋盘->相机) 外参 -> 转到 (世界->相机)
    results = {}
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    for key in ("front", "right", "back", "left"):
        print(f"Calib {key}")
        img_path = image_map[key]
        if not Path(img_path).is_file():
            raise typer.BadParameter(f"图像不存在: {key} -> {img_path}")

        camera = cam_map[key]
        # 使用图像所在目录作为 working_dir，避免 ExtCalibrationEngine 初始化期望目录下无图片
        engine = ExtCalibrationEngine(str(Path(img_path).parent), chessboard_size, key, square_size)
        ok = engine.detect_corners2(img_path, check=False)
        if not ok:
            raise RuntimeError(f"{key} 未检测到有效棋盘角点: {img_path}")

        # 外参标定得到 board坐标到camera坐标变换矩阵（非联合标定的单路外参）
        Rt_bc, rms_px = engine.extract_extrinsic(camera, depth_prior=depth_prior, depth_weight=depth_weight)
        R_bc = Rt_bc[:, :3]
        t_bc = Rt_bc[:, 3]
        r_bc, p_bc, y_bc = rpy_from_R(R_bc)
        with np.printoptions(precision=6, suppress=True):
            typer.echo(f"[{key}] 单路外参(棋盘->相机) [R|t]:")
            typer.echo(Rt_bc)
        typer.echo(f"[{key}] t (units of <square_size>): x={t_bc[0]:.6f}, y={t_bc[1]:.6f}, z={t_bc[2]:.6f}")
        typer.echo(f"[{key}] rpy (deg): roll={r_bc:.3f}, pitch={p_bc:.3f}, yaw={y_bc:.3f}; RMS={rms_px:.4f} px")

        # 可视化1（棋盘坐标轴，便于核验局部外参）：
        overlay_board = engine.visualize(camera, axis_length=axis_length)
        out_img = output_dir / f"{Path(img_path).stem}_axes_board_{key}.jpg"
        cv.imwrite(str(out_img), overlay_board)

        # 转换到世界坐标系
        R_wb, t_wb = Twb[key]
        R_wc, t_wc = compose_world_extrinsic(R_bc, t_bc, R_wb, t_wb)
        roll, pitch, yaw = rpy_from_R(R_wc)

        Rt_wc = np.hstack([R_wc, t_wc.reshape(3, 1)])
        # 同时计算 cam->world 外参：R_cw = R_wc^T, t_cw = -R_wc^T t_wc
        R_cw = R_wc.T
        t_cw = -R_cw @ t_wc
        r_cw, p_cw, y_cw = rpy_from_R(R_cw)
        Rt_cw = np.hstack([R_cw, t_cw.reshape(3, 1)])


        results[key] = {
            "rms_px": float(rms_px),
            "image_path": str(img_path),
            "board2cam": {
                "Rt": Rt_bc.tolist(),
                "xyz": t_bc.tolist(),
                "rpy_deg": [float(r_bc), float(p_bc), float(y_bc)],
            },
            "world2cam": {
                "Rt": Rt_wc.tolist(),
                "xyz": t_wc.tolist(),
                "rpy_deg": [float(roll), float(pitch), float(yaw)],
            },
            "cam2world": {
                "Rt": Rt_cw.tolist(),
                "xyz": t_cw.tolist(),
                "rpy_deg": [float(r_cw), float(p_cw), float(y_cw)],
            }
        }

        if show:
            cv.imshow(f"{key}-board", overlay_board)

        # 交互验证：拾取 8 个像素点，打印世界坐标（Z=0 平面交点）
        if verify:
            _pick_and_print_world_points(engine.image, camera, Rt_wc, count=4, win_name=f"pick-{key}")

        # 导出 TXT: 相机内参（输出到 checkpoints_dir，例如 outputs/assosicate）
        calib_txt_dir = checkpoints_dir
        calib_txt_dir.mkdir(parents=True, exist_ok=True)

        #cam_name = camera.name if getattr(camera, 'name', None) else key
        cam_name = key
        # 1) 内参 TXT（复用 ocam 格式字段）
        try:
            height, width = engine.image.shape[:2]
            direct_poly = np.asarray(camera.taylor_coefficient).ravel()
            inverse_poly = np.asarray(camera.inverse_poly).ravel()
            # revese the order of inverse_poly as the libxcam project needs
            inverse_poly_reversed = inverse_poly[::-1]
            center_col, center_row = float(camera.distortion_center[0]), float(camera.distortion_center[1])
            stretch = np.asarray(camera.stretch_matrix, dtype=float)
            c_param = float(stretch[0, 0])
            d_param = float(stretch[0, 1])
            e_param = float(stretch[1, 0])
            fx = c_param * (width / 2.0)
            fy = float(stretch[1, 1]) * (height / 2.0)
            intrinsic_matrix_line = f"{fx:.9g} 0 {center_col:.9g} 0 {fy:.9g} {center_row:.9g} 0 0 1 \n"

            def _format_coefficients(coeffs: np.ndarray) -> str:
                return f"{coeffs.shape[0]} " + " ".join(f"{v:.9g}" for v in coeffs) + " \n"

            intr_lines = [
                "\n#polynomial coefficients for the DIRECT mapping function (ocam_model.ss in MATLAB). These are used by cam2world\n\n",
                _format_coefficients(direct_poly),
                '#polynomial coefficients for the inverse mapping function (ocam_model.invpol in MATLAB). These are used by world2cam\n\n',
                _format_coefficients(inverse_poly_reversed),
                '\n#center: "row" and "column", starting from 0 (C convention)\n\n',
                f"{center_row:.9g} {center_col:.9g}\n\n",
                '#affine parameters "c", "d", "e"\n\n',
                f"{c_param:.9g} {d_param:.9g} {e_param:.9g}\n\n",
                '#image size: "height" and "width"\n\n',
                f"{height} {width}\n\n",
                '#camera Intrinsic parmeters: <fx 0 cx, 0 fy cy, 0 0 1>\n\n',
                intrinsic_matrix_line,
            ]
            if cam_name == "back":
                intr_path = calib_txt_dir / "intrinsic_camera_rear.txt"
            else:
                intr_path = calib_txt_dir / f"intrinsic_camera_{cam_name}.txt"
            with open(intr_path, 'w', encoding='utf-8') as f_txt:
                f_txt.writelines(intr_lines)
                print(f"Save intrinsic file {intr_path} for {cam_name}")
        except Exception as e:
            logger.warning(f"导出内参 TXT 失败 ({cam_name}): {e}")

        # 外参 TXT 在联合标定完成后统一按 cam2ego 写出

    # 弹窗展示
    if show:
        logger.info("按任意键关闭所有窗口...")
        cv.waitKey(0)
        for key in ("front", "right", "back", "left"):
            try:
                cv.destroyWindow(f"{key}-board")
            except Exception:
                pass

    # 计算 ego 坐标到世界坐标的变换，以及每个相机的 ego->cam 外参
    R_ew, t_ew = get_ego2world(results)
    R_ew = np.asarray(R_ew, dtype=np.float64).reshape(3, 3)
    t_ew = np.asarray(t_ew, dtype=np.float64).reshape(3)

    # 计算每个相机的 ego->cam 与 cam->ego，并导出 cam2ego 外参 TXT
    calib_txt_dir = checkpoints_dir
    calib_txt_dir.mkdir(parents=True, exist_ok=True)

    for key, cam_info in results.items():
        Rt_wc = np.asarray(cam_info["world2cam"]["Rt"], dtype=np.float64)
        R_wc = Rt_wc[:, :3]
        t_wc = Rt_wc[:, 3]

        # ego -> cam: X_c = R_wc (R_ew X_e + t_ew) + t_wc
        R_ec = R_wc @ R_ew
        t_ec = R_wc @ t_ew + t_wc
        Rt_ec = np.hstack([R_ec, t_ec.reshape(3, 1)])
        r_ec, p_ec, y_ec = rpy_from_R(R_ec)

        cam_info["ego2cam"] = {
            "Rt": Rt_ec.tolist(),
            "xyz": t_ec.tolist(),
            "rpy_deg": [float(r_ec), float(p_ec), float(y_ec)],
        }

        # cam -> ego: 取 ego->cam 的逆变换
        R_ce = R_ec.T
        t_ce = -R_ce @ t_ec
        Rt_ce = np.hstack([R_ce, t_ce.reshape(3, 1)])
        r_ce, p_ce, y_ce = rpy_from_R(R_ce)

        cam_info["cam2ego"] = {
            "Rt": Rt_ce.tolist(),
            "xyz": t_ce.tolist(),
            "rpy_deg": [float(r_ce), float(p_ce), float(y_ce)],
        }

        # 不需要CamB坐标系
        calc_camb = False
        if (calc_camb):
            # camB(cam_world)
            # ego -> camB: 先从 ego 到 cam，再从 cam 到 camB
            R_ccb, t_ccb = get_cam2camb()
            R_ecb = R_ccb @ R_ec
            t_ecb = R_ccb @ t_ec + t_ccb
            Rt_ecb = np.hstack([R_ecb, t_ecb.reshape(3, 1)])
            r_ecb, p_ecb, y_ecb = rpy_from_R(R_ecb)

            cam_info["ego2camB"] = {
                "Rt": Rt_ecb.tolist(),
                "xyz": t_ecb.tolist(),
                "rpy_deg": [float(r_ecb), float(p_ecb), float(y_ecb)],
            }

            # camB -> ego: 取 ego->camB 的逆变换
            R_cbe = R_ecb.T
            t_cbe = -R_cbe @ t_ecb
            Rt_cbe = np.hstack([R_cbe, t_cbe.reshape(3, 1)])
            r_cbe, p_cbe, y_cbe = rpy_from_R(R_cbe)

            cam_info["camB2ego"] = {
                "Rt": Rt_cbe.tolist(),
                "xyz": t_cbe.tolist(),
                "rpy_deg": [float(r_cbe), float(p_cbe), float(y_cbe)],
            }

        # 3) cam2ego 外参 TXT（描述相机在 ego 坐标中的位姿）
        print("Using cam2ego...")
        xyz = t_ce.tolist()
        rpy_deg = [float(r_ce), float(p_ce), float(y_ce)]
        save_extrinsic_txt(calib_txt_dir, key, xyz=xyz, rpy_deg=rpy_deg)


    # 保存联合外参 JSON
    ts = time.strftime("%Y%m%d_%H%M%S")
    extr_path = checkpoints_dir / f"avm_extrinsics_{ts}.json"
    with open(extr_path, "w", encoding="utf-8") as f:
        json.dump({
            "square_size": float(square_size),
            "chessboard_size": chessboard_size,
            "world2board": {
                k: {"R_wb": Twb[k][0].tolist(), "t_wb": Twb[k][1].tolist()} for k in ("front", "right", "back", "left")
            },
            "ego2world": {
                "R_ew": R_ew.tolist(),
                "t_ew": t_ew.tolist(),
            },
            "cameras": results,
        }, f, ensure_ascii=False, indent=2)
    typer.echo(f"联合外参已保存: {extr_path}")


if __name__ == "__main__":
    app()
