"""
OpenCV fisheye intrinsics calibration CLI.

Given a folder or glob of chessboard images, estimates K and D (k1..k4)
using cv2.fisheye.calibrate and saves results to a JSON file.
"""
from pathlib import Path
from typing import List, Tuple, Optional
import json
import glob

import cv2 as cv
import numpy as np
import typer

from pyocamcalib.modelling.utils import get_files


def _collect_images(source: str) -> List[Path]:
    p = Path(source)
    if p.is_dir():
        return sorted(get_files(p))
    # treat as glob
    paths = [Path(s) for s in glob.glob(source)]
    return sorted([pp for pp in paths if pp.is_file()])


def _detect_corners(image: np.ndarray,
                    pattern_size: Tuple[int, int],
                    use_sb: bool = True) -> Optional[np.ndarray]:
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    flags = cv.CALIB_CB_FAST_CHECK | cv.CALIB_CB_NORMALIZE_IMAGE | cv.CALIB_CB_ADAPTIVE_THRESH
    if use_sb and hasattr(cv, "findChessboardCornersSB"):
        ok, corners = cv.findChessboardCornersSB(gray, pattern_size)
        if not ok:
            ok, corners = cv.findChessboardCorners(gray, pattern_size, flags)
    else:
        ok, corners = cv.findChessboardCorners(gray, pattern_size, flags)
    if not ok:
        return None
    term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    cv.cornerSubPix(gray, corners, (5, 5), (-1, -1), term)
    return corners.reshape(-1, 1, 2)


def _build_object_points(pattern_size: Tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 1, 3), np.float64)
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    objp[:, 0, 0] = grid_x.reshape(-1) * square_size
    objp[:, 0, 1] = grid_y.reshape(-1) * square_size
    objp[:, 0, 2] = 0.0
    return objp


def _save_json(K: np.ndarray,
               D: np.ndarray,
               image_size: Tuple[int, int],
               rms: float,
               pattern_size: Tuple[int, int],
               square_size: float,
               camera_name: Optional[str],
               out_path: Path) -> None:
    out = {
        "model": "opencv_fisheye",
        "camera_name": camera_name,
        "image_size": [int(image_size[0]), int(image_size[1])],  # (w,h)
        "K": K.tolist(),
        "D": D.reshape(-1).tolist(),
        "rms": float(rms),
        "pattern_size": [int(pattern_size[0]), int(pattern_size[1])],
        "square_size": float(square_size),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)


def _per_view_rms(K: np.ndarray,
                  D: np.ndarray,
                  objpoints: List[np.ndarray],
                  imgpoints: List[np.ndarray],
                  rvecs: List[np.ndarray],
                  tvecs: List[np.ndarray]) -> List[float]:
    rms_list: List[float] = []
    for obj, imgp, rvec, tvec in zip(objpoints, imgpoints, rvecs, tvecs):
        proj, _ = cv.fisheye.projectPoints(obj, rvec, tvec, K, D)
        proj2 = proj.reshape(-1, 2)
        obs2 = imgp.reshape(-1, 2)
        err = obs2 - proj2
        per_view = float(np.sqrt((err[:, 0]**2 + err[:, 1]**2).mean()))
        rms_list.append(per_view)
    return rms_list

"""
python -m pyocamcalib.script.opencv_fisheye_calib_intr ./test_images/inhandus_1 \
        --chessboard-size-row 8 --chessboard-size-column 6 --square-size 1 --camera-name inhandus_1
"""

def main(
    images: str = typer.Argument(..., help="Folder path or glob for chessboard images"),
    camera_name: Optional[str] = typer.Option(None, help="Camera name for the output file."),
    chessboard_size_row: int = typer.Option(9, help="Number of inner corners along a row (columns in grid)."),
    chessboard_size_column: int = typer.Option(6, help="Number of inner corners along a column (rows in grid)."),
    square_size: float = typer.Option(30.0, help="Chessboard square size (mm or chosen units)."),
    use_sb: bool = typer.Option(True, help="Use findChessboardCornersSB if available."),
    output: Optional[Path] = typer.Option(None, help="Output JSON path (defaults to checkpoints/calibration)."),
    visualize: bool = typer.Option(True, help="Save per-image detection overlays."),
    show: bool = typer.Option(True, help="Show windows interactively (may not work headless)."),
    overlays_dir: Path = typer.Option(Path("./outputs/fisheye_intr_overlays"), help="Directory for overlay images."),
    auto_filter: bool = typer.Option(True, help="Automatically drop ill-conditioned/high-error views and recalibrate."),
    filter_factor: float = typer.Option(3.0, help="Drop views with per-view RMS > factor * median RMS."),
    max_recal: int = typer.Option(2, help="Max auto-filtering recalibration iterations."),
):
    pattern_size = (chessboard_size_row, chessboard_size_column)
    image_paths = _collect_images(images)
    if not image_paths:
        raise typer.BadParameter(f"No images found from: {images}")

    objp = _build_object_points(pattern_size, square_size)
    objpoints: List[np.ndarray] = []
    imgpoints: List[np.ndarray] = []
    img_size_wh = None
    used_paths: List[Path] = []

    used = 0
    for p in image_paths:
        img = cv.imread(str(p))
        if img is None:
            continue
        if img_size_wh is None:
            img_size_wh = (img.shape[1], img.shape[0])  # (w,h)
        corners = _detect_corners(img, pattern_size, use_sb=use_sb)
        if corners is None or corners.shape[0] != objp.shape[0]:
            continue
        objpoints.append(objp.copy())
        imgpoints.append(corners.copy())
        used_paths.append(p)
        used += 1

        if visualize:
            vis = img.copy()
            cv.drawChessboardCorners(vis, pattern_size, corners, True)
            overlays_dir.mkdir(parents=True, exist_ok=True)
            out_path = overlays_dir / f"{p.stem}_corners.jpg"
            cv.imwrite(str(out_path), vis)
            if show:
                cv.imshow("corners", vis)
                cv.waitKey(1)

    if used < 3:
        raise typer.BadParameter("Not enough valid views (need >= 3)")

    print(f"Detect corners: {used}/{len(image_paths)}")

    K = np.zeros((3, 3), dtype=np.float64)
    D = np.zeros((4, 1), dtype=np.float64)
    rvecs: List[np.ndarray] = []
    tvecs: List[np.ndarray] = []
    flags = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv.fisheye.CALIB_CHECK_COND | cv.fisheye.CALIB_FIX_SKEW
    criteria = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    # Calibrate with robust fallbacks and optional auto-filtering of bad views
    def _calibrate_with_flags(flags_try: int,
                              obj: List[np.ndarray],
                              imgp: List[np.ndarray]):
        return cv.fisheye.calibrate(
            objectPoints=obj,
            imagePoints=imgp,
            image_size=img_size_wh,
            K=K,
            D=D,
            rvecs=[],
            tvecs=[],
            flags=flags_try,
            criteria=criteria,
        )

    def _try_fallbacks(obj: List[np.ndarray], imgp: List[np.ndarray]):
        try:
            return _calibrate_with_flags(flags, obj, imgp), flags
        except cv.error:
            flags1 = cv.fisheye.CALIB_RECOMPUTE_EXTRINSIC | cv.fisheye.CALIB_FIX_SKEW
            try:
                return _calibrate_with_flags(flags1, obj, imgp), flags1
            except cv.error:
                flags2 = 0
                return _calibrate_with_flags(flags2, obj, imgp), flags2

    (rms, K, D, rvecs, tvecs), used_flags = _try_fallbacks(objpoints, imgpoints)

    if auto_filter and used > 6:
        for it in range(max_recal):
            per_view = _per_view_rms(K, D, objpoints, imgpoints, rvecs, tvecs)
            med = float(np.median(per_view))
            thr = max(med * filter_factor, med + 1.0)  # also allow absolute margin
            keep_idx = [i for i, v in enumerate(per_view) if v <= thr]
            drop = len(per_view) - len(keep_idx)
            if drop == 0 or len(keep_idx) < 3:
                break
            objpoints = [objpoints[i] for i in keep_idx]
            imgpoints = [imgpoints[i] for i in keep_idx]
            used_paths = [used_paths[i] for i in keep_idx]
            (rms, K, D, rvecs, tvecs), used_flags = _try_fallbacks(objpoints, imgpoints)
            typer.echo(f"Auto-filter iteration {it+1}: dropped {drop} views; new views={len(objpoints)}; RMS={rms:.4f}")

    if output is None:
        ts = cv.getTickCount()
        out_dir = Path("src/pyocamcalib/checkpoints/calibration")
        out_path = out_dir / (
            f"opencv_fisheye_intrinsics_{camera_name or 'cam'}_{ts}.json"
        )
    else:
        out_path = output

    _save_json(K, D, img_size_wh, rms, pattern_size, square_size, camera_name, out_path)

    typer.echo(f"Views used: {used}")
    typer.echo(f"Image size (w,h): {img_size_wh}")
    typer.echo(f"RMS: {rms:.4f} px")
    typer.echo(f"Flags used: {used_flags}")
    with np.printoptions(precision=6, suppress=True):
        typer.echo(f"K=\n{K}")
        typer.echo(f"D= {D.ravel().tolist()}")
    typer.echo(f"Saved: {out_path}")

    # Per-image report: view extrinsics and per-view RMS, plus reprojection overlay
    if visualize or show:
        for i, (p, obj, imgp) in enumerate(zip(used_paths, objpoints, imgpoints)):
            rvec_i = rvecs[i]
            tvec_i = tvecs[i]
            proj, _ = cv.fisheye.projectPoints(obj, rvec_i, tvec_i, K, D)
            proj2 = proj.reshape(-1, 2)
            obs2 = imgp.reshape(-1, 2)
            err = obs2 - proj2
            per_view_rms = float(np.sqrt((err[:, 0]**2 + err[:, 1]**2).mean()))

            # Save overlay image comparing observed vs projected
            vis = cv.imread(str(p))
            if vis is not None:
                for q in proj2:
                    cv.circle(vis, (int(round(q[0])), int(round(q[1]))), 3, (0, 255, 0), -1)
                for q in obs2:
                    cv.circle(vis, (int(round(q[0])), int(round(q[1]))), 2, (0, 0, 255), -1)
                text = f"RMS: {per_view_rms:.3f}px"
                cv.putText(vis, text, (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
                out_path_ov = overlays_dir / f"{p.stem}_reproj.jpg"
                cv.imwrite(str(out_path_ov), vis)
                if show:
                    cv.imshow("reproj", vis)
                    cv.waitKey(1)

            # Print per-view numbers
            with np.printoptions(precision=6, suppress=True):
                typer.echo(f"View {i+1}: {p.name}")
                typer.echo(f"  rvec: {rvec_i.reshape(-1)}")
                typer.echo(f"  tvec: {tvec_i.reshape(-1)}")
                typer.echo(f"  per-view RMS: {per_view_rms:.4f} px")

    if show:
        cv.waitKey(0)
        cv.destroyAllWindows()


if __name__ == "__main__":
    typer.run(main)
