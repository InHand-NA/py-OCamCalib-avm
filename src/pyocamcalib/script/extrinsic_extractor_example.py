#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import json
import glob
import math
from dataclasses import dataclass
from typing import Tuple, List, Dict

import cv2 as cv
import numpy as np
from scipy.optimize import least_squares


# ----------------------------
# OCamCalib model structures
# ----------------------------
@dataclass
class OCamModel:
    pol: np.ndarray      # polynomial coefficients: rho = pol(0) + pol(1)*m_z + pol(2)*m_z^2 + ...
    invpol: np.ndarray   # inverse polynomial: m_z = invpol(0) + invpol(1)*rho + invpol(2)*rho^2 + ...
    xc: float            # principal point x (column)
    yc: float            # principal point y (row)
    c: float             # affine param
    d: float             # affine param
    e: float             # affine param

    @staticmethod
    def from_dict(d: Dict) -> "OCamModel":
        return OCamModel(
            pol=np.array(d["pol"], dtype=np.float64),
            invpol=np.array(d["invpol"], dtype=np.float64),
            xc=float(d["xc"]),
            yc=float(d["yc"]),
            c=float(d["c"]),
            d=float(d["d"]),
            e=float(d["e"]),
        )


# ----------------------------
# OCamCalib projection funcs
# ----------------------------
def ocam_world2cam(m: np.ndarray, ocam: OCamModel) -> np.ndarray:
    """
    world2cam: from direction vector (unit) m=[mx,my,mz] in camera frame
    to pixel [u,v] using OCamCalib model.
    """
    mx, my, mz = m[..., 0], m[..., 1], m[..., 2]
    # rho = r(mz) using polynomial "pol"
    # r = pol[0] + pol[1]*mz + pol[2]*mz^2 + ...
    powers = np.vstack([mz**k for k in range(len(ocam.pol))]).T
    rho = powers @ ocam.pol

    # affine mapping to pixels
    u = mx * ocam.c + my * ocam.d + ocam.xc + 1e-16  # avoid zero division later
    v = mx * ocam.e + my         + ocam.yc

    # normalize so that sqrt(u'^2 + v'^2) equals rho, i.e., scale (u-xc,v-yc)
    du = u - ocam.xc
    dv = v - ocam.yc
    r_uv = np.sqrt(du*du + dv*dv) + 1e-16
    scale = rho / r_uv
    u_proj = ocam.xc + du * scale
    v_proj = ocam.yc + dv * scale

    return np.stack([u_proj, v_proj], axis=-1)


def ocam_cam2world(u: np.ndarray, ocam: OCamModel) -> np.ndarray:
    """
    cam2world: from pixel [u,v] to unit direction m=[mx,my,mz] in camera frame.
    """
    du = u[..., 0] - ocam.xc
    dv = u[..., 1] - ocam.yc

    # inverse affine
    denom = (ocam.c - ocam.d * ocam.e)
    mx = (du - ocam.d * dv) / denom
    my = (-ocam.e * du + ocam.c * dv) / denom

    rho = np.sqrt(mx*mx + my*my)
    # m_z = invpol[0] + invpol[1]*rho + invpol[2]*rho^2 + ...
    powers = np.vstack([rho**k for k in range(len(ocam.invpol))]).T
    mz = powers @ ocam.invpol

    # unnormalized direction then normalize
    m = np.stack([mx, my, mz], axis=-1)
    m = m / (np.linalg.norm(m, axis=-1, keepdims=True) + 1e-16)
    return m


# ----------------------------
# Geometry helpers
# ----------------------------
def rodrigues_to_R(rvec: np.ndarray) -> np.ndarray:
    R, _ = cv.Rodrigues(rvec.astype(np.float64))
    return R


def project_points_ocam(Pw: np.ndarray, rvec: np.ndarray, tvec: np.ndarray, ocam: OCamModel) -> np.ndarray:
    """
    Pw: (N,3) points in board/world frame (e.g., Z=0 if chessboard plane)
    rvec: (3,), tvec: (3,)
    returns pixels (N,2)
    """
    R = rodrigues_to_R(rvec)
    Xc = (Pw @ R.T) + tvec[None, :]
    # convert to unit direction
    Xc_norm = Xc / (np.linalg.norm(Xc, axis=1, keepdims=True) + 1e-16)
    uv = ocam_world2cam(Xc_norm, ocam)
    return uv


def residuals_pose(params: np.ndarray, Pw: np.ndarray, uv_obs: np.ndarray, ocam: OCamModel) -> np.ndarray:
    rvec = params[0:3]
    tvec = params[3:6]
    uv_pred = project_points_ocam(Pw, rvec, tvec, ocam)
    return (uv_pred - uv_obs).ravel()


def solve_pose_ocam(Pw: np.ndarray, uv: np.ndarray, ocam: OCamModel,
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


# ----------------------------
# Chessboard detection
# ----------------------------
def detect_chessboard_corners(img_gray: np.ndarray, pattern_size: Tuple[int, int]) -> np.ndarray:
    """
    pattern_size: (cols, rows) inner corners, e.g., (9,6)
    returns subpixel corners of shape (N,2) in (u,v) = (x,y) order
    """
    flags = cv.CALIB_CB_ADAPTIVE_THRESH + cv.CALIB_CB_NORMALIZE_IMAGE + cv.CALIB_CB_FAST_CHECK
    ret, corners = cv.findChessboardCorners(img_gray, pattern_size, flags)
    if not ret:
        raise RuntimeError("Chessboard not found.")
    # Subpixel refine
    term = (cv.TERM_CRITERIA_EPS + cv.TERM_CRITERIA_MAX_ITER, 100, 1e-4)
    corners = cv.cornerSubPix(img_gray, corners, (9, 9), (-1, -1), term)
    return corners.reshape(-1, 2)


def build_board_points(pattern_size: Tuple[int, int], square_size: float) -> np.ndarray:
    cols, rows = pattern_size
    objp = np.zeros((rows * cols, 3), np.float64)
    grid_x, grid_y = np.meshgrid(np.arange(cols), np.arange(rows))
    objp[:, 0] = grid_x.flatten() * square_size
    objp[:, 1] = grid_y.flatten() * square_size
    objp[:, 2] = 0.0
    return objp


# ----------------------------
# Batch calibrate
# ----------------------------
def calibrate_extrinsics_for_images(
    image_glob: str,
    ocam_json: str,
    pattern_size: Tuple[int, int],
    square_size: float,
    visualize: bool = False
):
    with open(ocam_json, "r") as f:
        ocam = OCamModel.from_dict(json.load(f))

    images = sorted(glob.glob(image_glob))
    if len(images) == 0:
        raise FileNotFoundError(f"No images matched: {image_glob}")

    Pw = build_board_points(pattern_size, square_size)

    results = []
    for path in images:
        img = cv.imread(path, cv.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARN] Failed to read {path}, skip.")
            continue

        try:
            uv = detect_chessboard_corners(img, pattern_size)
        except RuntimeError as e:
            print(f"[WARN] {path}: {e}")
            continue

        # Initial guess:
        # - Estimate a homography from 2D grid to image (for a crude init),
        #   then set small rotation and put board ~1m away. It's ok to use zeros,
        #   LM will converge as long as the board is visible roughly in front.
        r0 = np.zeros(3, dtype=np.float64)
        t0 = np.array([0.0, 0.0, 1.0], dtype=np.float64)

        rvec, tvec, rms = solve_pose_ocam(Pw, uv, ocam, r0, t0)

        # Optional visualization: draw reprojections
        if visualize:
            uv_fit = project_points_ocam(Pw, rvec, tvec, ocam)
            vis = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
            for p in uv_fit:
                cv.circle(vis, (int(round(p[0])), int(round(p[1]))), 3, (0, 255, 0), -1)
            for p in uv:
                cv.circle(vis, (int(round(p[0])), int(round(p[1]))), 2, (0, 0, 255), -1)
            cv.putText(vis, f"RMS: {rms:.3f}px", (20, 40), cv.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
            cv.imshow("reprojection", vis)
            cv.waitKey(1)

        # Build 4x4 pose matrix [R|t]
        R = rodrigues_to_R(rvec)
        T = np.eye(4, dtype=np.float64)
        T[:3, :3] = R
        T[:3, 3] = tvec

        results.append({
            "image": path,
            "rvec": rvec.tolist(),
            "tvec": tvec.tolist(),
            "R": R.tolist(),
            "T_cam_from_board": T.tolist(),
            "rms_px": float(rms),
        })

        print(f"[OK] {path}: rms={rms:.3f}px")

    if visualize:
        cv.destroyAllWindows()

    # Save results
    out_json = "extrinsics_ocam_results.json"
    with open(out_json, "w") as f:
        json.dump({"results": results}, f, indent=2)
    print(f"\nSaved: {out_json}")
    return results


# ----------------------------
# CLI
# ----------------------------
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="OCamCalib-based extrinsic calibration with chessboard.")
    parser.add_argument("--images", required=True, help='Glob for chessboard images, e.g. "data/*.png"')
    parser.add_argument("--ocam", required=True, help="OCamCalib intrinsics JSON file")
    parser.add_argument("--pattern", required=True, help="Inner corners as CxR, e.g. 9x6")
    parser.add_argument("--square", required=True, type=float, help="Square size in meters, e.g. 0.025")
    parser.add_argument("--vis", action="store_true", help="Visualize reprojection overlay")
    args = parser.parse_args()

    cols, rows = map(int, args.pattern.lower().split("x"))
    calibrate_extrinsics_for_images(
        args.images, args.ocam, (cols, rows), args.square, visualize=args.vis
    )
