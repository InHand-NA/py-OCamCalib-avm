import cv2
import numpy as np
import glob
import json
import os


def _try_calibrate(objpoints, imgpoints, image_size, flags, criteria):
    """Helper: run fisheye.calibrate with safe initial K/D and return outputs."""
    K_try = np.eye(3, 3, dtype=np.float64)
    D_try = np.zeros((4, 1), dtype=np.float64)
    return cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        image_size,
        K_try,
        D_try,
        None,
        None,
        flags,
        criteria,
    )


def diagnose_bad_frames(objpoints, imgpoints, image_size, flags, criteria, filenames):
    """Print which frame index/filename likely causes ill-conditioning.

    Strategy:
    1) Incrementally add frames until calibration fails; record the failing index.
    2) Leave-one-out within the failing prefix; any index that, when removed, restores success is a suspect.
    """
    print("[diagnose] Starting incremental calibration...")
    n_min = 2 #3
    N = len(objpoints)
    if N < n_min:
        print(f"[diagnose] Not enough views (have {N}, need >= {n_min}).")
        return

    fail_n = None
    for n in range(n_min, N + 1):
        try:
            _try_calibrate(objpoints[:n], imgpoints[:n], image_size, flags, criteria)
        except cv2.error as e:
            fail_n = n
            bad_file = filenames[n-1] if n-1 < len(filenames) else f"index {n-1}"
            print(f"[diagnose] Failure when adding index {n-1} (1-based #{n}): {bad_file}")
            print(f"[diagnose] OpenCV error: {e}")
            break

    if fail_n is None:
        print("[diagnose] No incremental failure; degeneracy may involve multiple frames together.")
        return

    suspects = []
    prefix = fail_n
    print("[diagnose] Leave-one-out on failing prefix to find culprit(s)...")
    for i in range(prefix):
        try:
            sel_obj = objpoints[:prefix]
            sel_img = imgpoints[:prefix]
            sel_obj = sel_obj[:i] + sel_obj[i+1:]
            sel_img = sel_img[:i] + sel_img[i+1:]
            _try_calibrate(sel_obj, sel_img, image_size, flags, criteria)
            suspects.append(i)
        except cv2.error:
            pass

    if suspects:
        print("[diagnose] Suspect frames (removing this frame succeeds):")
        for idx in suspects:
            name = filenames[idx] if idx < len(filenames) else f"index {idx}"
            print(f"  - index {idx} (1-based #{idx+1}): {name}")
    else:
        print("[diagnose] No single-frame culprit; multiple frames jointly ill-conditioned.")


def get_K_and_D(checkerboard, imgsPath):

    CHECKERBOARD = checkerboard
    subpix_criteria = (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1)
    #calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_CHECK_COND+cv2.fisheye.CALIB_FIX_SKEW
    calibration_flags = cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC+cv2.fisheye.CALIB_FIX_SKEW
    objp = np.zeros((1, CHECKERBOARD[0]*CHECKERBOARD[1], 3), np.float32)
    objp[0,:,:2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2)
    _img_shape = None
    objpoints = []
    imgpoints = []
    images = glob.glob(imgsPath + '/*.jpg')
    used_files = []
    corners_dir = os.path.join('.', 'corners')
    os.makedirs(corners_dir, exist_ok=True)
    used = 0
    for fname in images:
        print(f"file: {fname}")
        img = cv2.imread(fname)
        if _img_shape == None:
            _img_shape = img.shape[:2]
        else:
            assert _img_shape == img.shape[:2], "All images must share the same size."

        gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, CHECKERBOARD,cv2.CALIB_CB_ADAPTIVE_THRESH+cv2.CALIB_CB_FAST_CHECK+cv2.CALIB_CB_NORMALIZE_IMAGE)
        if ret == True:
            cv2.cornerSubPix(gray,corners,(3,3),(-1,-1),subpix_criteria)
            # Always save a visualization with drawn corners for diagnostics
            img_draw = img.copy()
            cv2.drawChessboardCorners(img_draw, CHECKERBOARD, corners, True)
            out_name = f"corners_{os.path.basename(fname)}"
            out_path = os.path.join(corners_dir, out_name)
            cv2.imwrite(out_path, img_draw)

            # Optional: skip frames with very low spatial spread to avoid degeneracy
            if np.min(np.std(corners.reshape(-1,2), axis=0)) < 5:
                print(f"[check] Skip low-spread corners: {fname}")
                continue

            # Use copies and consistent shapes for calibration
            objpoints.append(objp.copy())
            corners = corners.reshape(1, -1, 2)
            imgpoints.append(corners)
            used_files.append(fname)
            used += 1
        else:
            print(f"Fail to detect corner: {fname}")

    print(f"Detect corners: {used}/{len(images)}")
    N_OK = len(objpoints)
    K = np.zeros((3, 3))
    D = np.zeros((4, 1))
    rvecs = []
    tvecs = []
    try:
        rms, _, _, _, _ = cv2.fisheye.calibrate(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            K,
            D,
            rvecs,
            tvecs,
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
        )
    except cv2.error as e:
        print("[error] fisheye.calibrate failed. Running diagnostics...")
        diagnose_bad_frames(
            objpoints,
            imgpoints,
            gray.shape[::-1],
            calibration_flags,
            (cv2.TERM_CRITERIA_EPS+cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-6),
            used_files,
        )
        raise
    DIM = _img_shape[::-1]
    print("Found " + str(N_OK) + " valid images for calibration")
    print("DIM=" + str(_img_shape[::-1]))
    print("K=np.array(" + str(K.tolist()) + ")")
    print("D=np.array(" + str(D.tolist()) + ")")
    print(f"RMS: {rms}")
    return DIM, K, D


def undistort(img_path,K,D,DIM,scale=1.0, imshow=False):
    img = cv2.imread(img_path)
    dim1 = img.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"
    if dim1[0]!=DIM[0]:
        img = cv2.resize(img,DIM,interpolation=cv2.INTER_AREA)
    Knew = K.copy()
    if scale:#change fov
        Knew[(0,1), (0,1)] = scale * Knew[(0,1), (0,1)]
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(K, D, np.eye(3), Knew, DIM, cv2.CV_16SC2)
    undistorted_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    if imshow:
        cv2.imshow("undistorted", undistorted_img)
    return undistorted_img

def get_undistor_reproject_err(img_dir, K, D,  chessboardsize, square_size, k_scale=1.0):
    max_err_img_path = None
    images = sorted(glob.glob(os.path.join(img_dir, '*.jpg')))
    if not images:
        print(f"[eval] No images found in: {img_dir}")
        return None

    # Prepare object points once
    cols, rows = chessboardsize
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    # Build scaled intrinsics for evaluation (controls post-undistort K)
    K_eval = K.copy().astype(np.float64)
    if k_scale is not None and k_scale != 1.0:
        K_eval[0, 0] *= float(k_scale)
        K_eval[1, 1] *= float(k_scale)

    # Precompute undistort maps for speed
    first = cv2.imread(images[0])
    if first is None or first.size == 0:
        print(f"[eval] Failed to read first image: {images[0]}")
        return None
    h, w = first.shape[:2]
    DIM = (w, h)
    map1, map2 = cv2.fisheye.initUndistortRectifyMap(
        K, D, np.eye(3), K_eval, DIM, cv2.CV_16SC2
    )
    out_dir = os.path.join(img_dir, 'undist_eval')
    os.makedirs(out_dir, exist_ok=True)

    # Fast detection flags and sub-pixel refinement criteria
    det_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_FAST_CHECK
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    subpix_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1
    )

    errs = []
    max_err_val = -1.0
    for fp in images:
        img = cv2.imread(fp)
        if img is None or img.size == 0:
            print(f"[eval][skip] read fail: {fp}")
            continue
        if img.shape[1] != w or img.shape[0] != h:
            img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)

        # Save undistorted image (precomputed maps)
        undist_img = cv2.remap(img, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        cv2.imwrite(os.path.join(out_dir, os.path.basename(fp)), undist_img)

        # Corner detection and reprojection error on undistorted points
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, chessboardsize, det_flags)
        if not found:
            print(f"[eval][skip] chessboard not found: {fp}")
            continue
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)

        # Undistort detected corners to pinhole model with K
        undist_corners = cv2.fisheye.undistortPoints(corners, K, D, P=K_eval).reshape(-1, 2).astype(np.float32)

        # Solve pose; prefer IPPE for planar targets, fallback to ITERATIVE
        try:
            ok, rvec, tvec = cv2.solvePnP(
                objp, undist_corners, K_eval, None, flags=cv2.SOLVEPNP_IPPE
            )
        except cv2.error:
            ok, rvec, tvec = cv2.solvePnP(
                objp, undist_corners, K_eval, None, flags=cv2.SOLVEPNP_ITERATIVE
            )
        if not ok:
            print(f"[eval][skip] solvePnP failed: {fp}")
            continue

        proj, _ = cv2.projectPoints(objp, rvec, tvec, K_eval, None)
        proj2 = proj.reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum((proj2 - undist_corners) ** 2, axis=1))))
        errs.append(err)
        if err > max_err_val:
            max_err_val = err
            max_err_img_path = fp

    if not errs:
        print("[eval] No valid frames for error evaluation.")
        return None

    min_err = float(np.min(errs))
    max_err = float(np.max(errs))
    mean_err = float(np.mean(errs))
    print(f"image count: {len(images)}, err count: {len(errs)}  k_scale={k_scale}")
    if max_err_img_path is not None:
        print(f"[eval] max_err image: {max_err_img_path}")
    print(f"[eval] Reprojection error px  count={len(errs)}  min={min_err:.4f}  max={max_err:.4f}  mean={mean_err:.4f}")
    return {"min": min_err, "max": max_err, "mean": mean_err, "count": len(errs)}

def extrinsic_calibration_fisheye_fixed_kd(
    img_file_path,
    K,
    D,
    chessboardsize,
    square_size,
    prior_rvec=None,
    prior_tvec=None,
    weight_rot=0.0,
    weight_t=0.0,
):
    """Estimate per-image extrinsics (rvec/tvec) with fisheye.calibrate fixing K,D.

    - Uses provided K, D as fixed intrinsics and optimizes only extrinsics.
    - Optional weak prior: set `prior_rvec` and/or `prior_tvec` with weights
      (`weight_rot` in deg per unit cost; `weight_t` scalar or 3-vector). A
      lightweight post-selection via PnP candidates nudges the solution toward
      priors without hard constraints.
    - Returns a dict with RMS, rvecs, tvecs, and the filenames used.
    """
    images = [img_file_path]

    cols, rows = chessboardsize
    # Object points in board frame (single template reused per image)
    objp = np.zeros((1, cols * rows, 3), np.float32)
    objp[0, :, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    # Detection settings
    det_flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_FAST_CHECK
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    subpix_criteria = (
        cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1
    )

    objpoints = []
    imgpoints = []
    used_files = []
    img_size = None

    for fp in images:
        img = cv2.imread(fp)
        if img is None or img.size == 0:
            continue
        if img_size is None:
            img_size = img.shape[1], img.shape[0]  # (w, h)
        else:
            if (img.shape[1], img.shape[0]) != img_size:
                img = cv2.resize(img, img_size, interpolation=cv2.INTER_AREA)

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        found, corners = cv2.findChessboardCorners(gray, chessboardsize, det_flags)
        if not found:
            continue
        cv2.cornerSubPix(gray, corners, (3, 3), (-1, -1), subpix_criteria)
        objpoints.append(objp.copy())
        imgpoints.append(corners.reshape(1, -1, 2))
        used_files.append(fp)

    if not objpoints:
        raise RuntimeError("No valid chessboard detections; cannot calibrate extrinsics.")

    K_init = np.asarray(K, dtype=np.float64).copy()
    D_init = np.asarray(D, dtype=np.float64).copy()

    # Build flags to fix intrinsics (K, D) and only solve extrinsics
    flags = 0
    # Always use intrinsic guess from provided K/D
    if hasattr(cv2.fisheye, 'CALIB_USE_INTRINSIC_GUESS'):
        flags |= cv2.fisheye.CALIB_USE_INTRINSIC_GUESS
    # Prefer recomputing extrinsic each iteration for stability
    if hasattr(cv2.fisheye, 'CALIB_RECOMPUTE_EXTRINSIC'):
        flags |= cv2.fisheye.CALIB_RECOMPUTE_EXTRINSIC
    # Fix intrinsics as much as the API exposes across OpenCV versions
    for name in (
        'CALIB_FIX_INTRINSIC',
        'CALIB_FIX_PRINCIPAL_POINT',
        'CALIB_FIX_FOCAL_LENGTH',
        'CALIB_FIX_SKEW',
        'CALIB_FIX_K1', 'CALIB_FIX_K2', 'CALIB_FIX_K3', 'CALIB_FIX_K4',
    ):
        if hasattr(cv2.fisheye, name):
            flags |= getattr(cv2.fisheye, name)

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)

    # Run calibration with fixed intrinsics to get rvecs/tvecs per frame
    rms, K_out, D_out, rvecs, tvecs = cv2.fisheye.calibrate(
        objpoints,
        imgpoints,
        img_size,
        K_init,
        D_init,
        None,
        None,
        flags,
        criteria,
    )

    # Ensure mutability: OpenCV may return tuples; convert to lists for edits
    try:
        rvecs = list(rvecs)
        tvecs = list(tvecs)
    except TypeError:
        pass

    print(f"[extrinsic] frames={len(objpoints)}  img_size={img_size}  RMS={float(rms):.6f}")
    # Summarize first few results for quick inspection
    preview = min(3, len(rvecs))
    for i in range(preview):
        rv = np.asarray(rvecs[i]).reshape(3)
        tv = np.asarray(tvecs[i]).reshape(3)
        print(f"  view[{i}] file={os.path.basename(used_files[i])}  rvec={rv}  tvec={tv}")

    # Optional weak-prior selection (single image): prefer PnP candidate closest to priors
    # Robustly detect nonzero weights for both scalar and vector inputs
    nonzero_rot = (float(weight_rot) != 0.0)
    try:
        wt_arr = np.asarray(weight_t, dtype=np.float64)
        nonzero_t = bool(np.any(wt_arr != 0.0))
    except Exception:
        nonzero_t = (float(weight_t) != 0.0)
    use_priors = (prior_rvec is not None or prior_tvec is not None) and (nonzero_rot or nonzero_t)
    if use_priors and len(imgpoints) == 1:
        def rot_angle_deg(rvec_a, rvec_b):
            Ra, _ = cv2.Rodrigues(np.asarray(rvec_a, dtype=np.float64).reshape(3, 1))
            Rb, _ = cv2.Rodrigues(np.asarray(rvec_b, dtype=np.float64).reshape(3, 1))
            M = Ra.T @ Rb
            tr = float(np.trace(M))
            c = max(-1.0, min(1.0, (tr - 1.0) * 0.5))
            return float(np.degrees(np.arccos(c)))

        prior_rvec = np.asarray(prior_rvec, dtype=np.float64).reshape(3, 1) if prior_rvec is not None else None
        prior_tvec = np.asarray(prior_tvec, dtype=np.float64).reshape(3, 1) if prior_tvec is not None else None
        if isinstance(weight_t, (list, tuple, np.ndarray)):
            w_t_vec = np.asarray(weight_t, dtype=np.float64).reshape(3,)
        else:
            w_t_vec = np.array([float(weight_t)] * 3, dtype=np.float64)

        # Undistorted points for pinhole model (K, None)
        und = cv2.fisheye.undistortPoints(imgpoints[0], K_init, D_init, P=K_init).reshape(-1, 2).astype(np.float32)
        obj = objpoints[0].reshape(-1, 3).astype(np.float32)

        # Baseline: fisheye.calibrate output
        r0 = np.asarray(rvecs[0], dtype=np.float64).reshape(3, 1)
        t0 = np.asarray(tvecs[0], dtype=np.float64).reshape(3, 1)
        proj0, _ = cv2.projectPoints(obj, r0, t0, K_init, None)
        e_reproj0 = float(np.sqrt(np.mean(np.sum((proj0.reshape(-1, 2) - und) ** 2, axis=1))))
        e_rot0 = rot_angle_deg(prior_rvec, r0) if prior_rvec is not None else 0.0
        e_t0 = float(np.linalg.norm((t0.reshape(3,) - prior_tvec.reshape(3,)) * w_t_vec)) if prior_tvec is not None else 0.0
        cost0 = e_reproj0 + float(weight_rot) * e_rot0 + e_t0

        # Candidates from PnPGeneric
        try:
            out = cv2.solvePnPGeneric(obj, und, K_init, None, flags=cv2.SOLVEPNP_IPPE)
        except cv2.error:
            out = cv2.solvePnPGeneric(obj, und, K_init, None, flags=cv2.SOLVEPNP_ITERATIVE)

        if len(out) == 4:
            _, r_list, t_list, _ = out
        elif len(out) == 3:
            r_list, t_list, _ = out
        else:
            r_list, t_list = out[0], out[1]

        best = (cost0, r0, t0)
        for rk, tk in zip(r_list, t_list):
            rk = np.asarray(rk, dtype=np.float64).reshape(3, 1)
            tk = np.asarray(tk, dtype=np.float64).reshape(3, 1)
            projk, _ = cv2.projectPoints(obj, rk, tk, K_init, None)
            e_reproj = float(np.sqrt(np.mean(np.sum((projk.reshape(-1, 2) - und) ** 2, axis=1))))
            e_rot = rot_angle_deg(prior_rvec, rk) if prior_rvec is not None else 0.0
            e_t = float(np.linalg.norm((tk.reshape(3,) - prior_tvec.reshape(3,)) * w_t_vec)) if prior_tvec is not None else 0.0
            cost = e_reproj + float(weight_rot) * e_rot + e_t
            if cost < best[0]:
                best = (cost, rk, tk)

        rvecs[0] = best[1]
        tvecs[0] = best[2]


    return {
        'rms': float(rms),
        'K': K_out,
        'D': D_out,
        'rvecs': rvecs,
        'tvecs': tvecs,
        'files': used_files,
        'image_size': img_size,
    }

def extrinsic_calibration2(img, K, D, chessboardsize, square_size):
    # Estimate extrinsic pose using cv2.solvePnPGeneric and let user pick solution.
    if img is None or img.size == 0:
        raise ValueError("Invalid input image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Detect corners
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_FAST_CHECK
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    found, corners = cv2.findChessboardCorners(gray, chessboardsize, flags)
    if not found:
        raise RuntimeError("Chessboard corners not found")

    cv2.cornerSubPix(
        gray,
        corners,
        (3, 3),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
    )

    # Build object points
    cols, rows = chessboardsize
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    # Undistort image points to match pinhole model with K, pass None distortion
    undist_corners = cv2.fisheye.undistortPoints(corners, K, D, P=K)
    undist_corners2 = undist_corners.reshape(-1, 2).astype(np.float32)

    # Solve PnP (generic) to get multiple candidates; prefer IPPE for planar boards
    rvecs_list = []
    tvecs_list = []
    try:
        out = cv2.solvePnPGeneric(
            objp.astype(np.float32),
            undist_corners2,
            K,
            None,
            flags=cv2.SOLVEPNP_IPPE,
        )
        # Tolerate OpenCV API variants: some return (retval, rvecs, tvecs, errs), others omit retval
        if len(out) == 4:
            _, rvecs_list, tvecs_list, _ = out
        elif len(out) == 3:
            rvecs_list, tvecs_list, _ = out
        else:
            # Fallback: try interpreting as (rvecs, tvecs)
            rvecs_list, tvecs_list = out[0], out[1]
    except cv2.error:
        # Fallback to ITERATIVE as a backup if IPPE unsupported
        out = cv2.solvePnPGeneric(
            objp.astype(np.float32),
            undist_corners2,
            K,
            None,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if len(out) == 4:
            _, rvecs_list, tvecs_list, _ = out
        elif len(out) == 3:
            rvecs_list, tvecs_list, _ = out
        else:
            rvecs_list, tvecs_list = out[0], out[1]

    # Normalize to Python lists
    if isinstance(rvecs_list, (np.ndarray, tuple)) and rvecs_list is not None and rvecs_list != []:
        pass
    # Safety: ensure we have iterable candidates
    try:
        n_candidates = len(rvecs_list)
    except TypeError:
        rvecs_list = [rvecs_list]
        tvecs_list = [tvecs_list]
        n_candidates = len(rvecs_list)

    if n_candidates == 0:
        raise RuntimeError("solvePnPGeneric returned no solutions")

    def draw_axes(base_img, rvec, tvec):
        vis = base_img.copy()
        cv2.drawChessboardCorners(vis, chessboardsize, corners, True)
        axis_len = float(square_size) * 3.0
        axis_obj = np.array(
            [
                [0.0, 0.0, 0.0],
                [axis_len, 0.0, 0.0],
                [0.0, axis_len, 0.0],
                [0.0, 0.0, -axis_len],
            ],
            dtype=np.float32,
        )
        axis_img, _ = cv2.fisheye.projectPoints(axis_obj.reshape(1, -1, 3), rvec, tvec, K, D)
        axis_img = axis_img.reshape(-1, 2).astype(int)
        o, x, y, z = axis_img
        cv2.line(vis, tuple(o), tuple(x), (0, 0, 255), 2)
        cv2.line(vis, tuple(o), tuple(y), (0, 255, 0), 2)
        cv2.line(vis, tuple(o), tuple(z), (255, 0, 0), 2)

        # Label axes X (red), Y (green), Z (blue)
        def put_axis_label(p_end, p_origin, text, color):
            v = (p_end - p_origin).astype(np.float32)
            n = float(np.hypot(v[0], v[1]))
            # place label slightly beyond the axis endpoint
            off = (v / (n if n > 1.0 else 1.0)) * 12.0
            pt = (int(p_end[0] + off[0]), int(p_end[1] + off[1]))
            cv2.putText(vis, text, pt, cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        put_axis_label(x, o, 'X', (0, 0, 255))
        put_axis_label(y, o, 'Y', (0, 255, 0))
        put_axis_label(z, o, 'Z', (255, 0, 0))
        return vis

    # Build candidate list with error and preview image
    candidates = []
    for i in range(n_candidates):
        rvec = np.asarray(rvecs_list[i], dtype=np.float64).reshape(3, 1)
        tvec = np.asarray(tvecs_list[i], dtype=np.float64).reshape(3, 1)
        proj, _ = cv2.projectPoints(objp, rvec, tvec, K, None)
        proj2 = proj.reshape(-1, 2)
        err = float(np.sqrt(np.mean(np.sum((proj2 - undist_corners2) ** 2, axis=1))))
        vis = draw_axes(img, rvec, tvec)
        # Annotate with index and error
        label = f"Candidate {i+1}/{n_candidates}  err={err:.3f} px"
        cv2.putText(vis, label, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(vis, "[a/d] prev/next  [space/enter] select  [q] auto-min", (10, 48), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)
        candidates.append({
            "rvec": rvec,
            "tvec": tvec,
            "error": err,
            "vis": vis,
        })

    # Start from min-error candidate
    best_idx = int(np.argmin([c["error"] for c in candidates]))
    idx = best_idx

    # Interactive selection
    win = 'pnp_candidates'
    cv2.imshow(win, candidates[idx]["vis"])
    selected = None
    while True:
        k = cv2.waitKey(0) & 0xFF
        if k in (ord('d'), 83):  # next (also handle right arrow code 83)
            idx = (idx + 1) % n_candidates
            cv2.imshow(win, candidates[idx]["vis"])
        elif k in (ord('a'), 81):  # prev (left arrow 81)
            idx = (idx - 1 + n_candidates) % n_candidates
            cv2.imshow(win, candidates[idx]["vis"])
        elif k in (13, 10, 32):  # enter or space
            selected = idx
            cv2.imwrite('ext_corner.jpg', candidates[idx]["vis"])
            break
        elif k == ord('q'):
            selected = best_idx
            break
        elif ord('1') <= k <= ord('9'):
            choice = k - ord('1')
            if 0 <= choice < n_candidates:
                selected = choice
                break
    cv2.destroyWindow(win)

    if selected is None:
        selected = best_idx

    sel = candidates[selected]
    rvec = sel["rvec"]
    tvec = sel["tvec"]
    err = sel["error"]

    # Compute RPY for selected
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))
    else:
        roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = 0.0

    result = {
        "rvec": rvec,
        "tvec": tvec,
        "txyz": (float(tvec[0]), float(tvec[1]), float(tvec[2])),
        "rpy": (float(roll), float(pitch), float(yaw)),
        "reproj_error": float(err),
        "annotated": sel["vis"],
    }
    print(f"Selected candidate {selected+1}/{n_candidates}: error={err:.3f} px")
    print(f"{result}")
    return result, rvec, tvec

def eval_extrinsic(img, K, D, rvec, tvec):
    if img is None or img.size == 0:
        raise ValueError("Invalid input image")

    K = np.asarray(K, dtype=np.float64)
    D = np.asarray(D, dtype=np.float64)
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3, 1)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3, 1)

    h, w = img.shape[:2]
    window = 'eval_extrinsic'
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)

    # Selection/display state
    clicks = []              # original image pixel coords
    scale = 1.0              # zoom factor
    hover_disp = None        # current mouse position in display coords
    annotated_img_base = img.copy()  # draw clicks/labels on original-size image

    def recompute_scaled_cache():
        if scale == 1.0:
            return annotated_img_base.copy()
        new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
        return cv2.resize(annotated_img_base, new_size, interpolation=cv2.INTER_NEAREST)

    scaled_cache = recompute_scaled_cache()

    def refresh_display():
        disp = scaled_cache.copy()
        if hover_disp is not None:
            x, y = hover_disp
            x = int(np.clip(x, 0, disp.shape[1]-1))
            y = int(np.clip(y, 0, disp.shape[0]-1))
            L = 12
            color = (0, 255, 255)
            cv2.line(disp, (x - L, y), (x + L, y), color, 1, cv2.LINE_AA)
            cv2.line(disp, (x, y - L), (x, y + L), color, 1, cv2.LINE_AA)
            cv2.circle(disp, (x, y), 1, color, -1, cv2.LINE_AA)
        hud = f"scale: {scale:.2f}x  points: {len(clicks)}/4  (Ctrl+= zoom in, Ctrl+- zoom out, ESC exit)"
        cv2.rectangle(disp, (5, 5), (5 + 520, 25), (0, 0, 0), -1)
        cv2.putText(disp, hud, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)
        cv2.imshow(window, disp)

    def disp_to_orig(pt):
        x, y = pt
        xo = int(round(x / scale))
        yo = int(round(y / scale))
        return int(np.clip(xo, 0, w - 1)), int(np.clip(yo, 0, h - 1))

    def on_mouse(event, x, y, flags, param):
        nonlocal hover_disp, scaled_cache
        hover_disp = (x, y)
        if event == cv2.EVENT_LBUTTONDOWN and len(clicks) < 8:
            xo, yo = disp_to_orig((x, y))
            clicks.append((xo, yo))
            cv2.circle(annotated_img_base, (xo, yo), 4, (0, 255, 255), -1)
            cv2.putText(annotated_img_base, f"({xo},{yo})", (xo + 6, yo - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
            scaled_cache = recompute_scaled_cache()
        refresh_display()

    cv2.setMouseCallback(window, on_mouse)
    refresh_display()

    # Loop: wait for 4 clicks or ESC; support zoom with Ctrl+= / Ctrl+- (or '='/'-')
    while True:
        key = cv2.waitKey(20) & 0xFF
        if len(clicks) >= 8:
            break
        if key == 27:
            break
        if key == ord('0'):
            scale = 1.0
            scaled_cache = recompute_scaled_cache()
            refresh_display()

    cv2.setMouseCallback(window, lambda *args: None)

    if len(clicks) == 0:
        print('[eval_extrinsic] No points selected.')
        cv2.destroyWindow(window)
        return []

    # Prepare for back-projection onto Z=0 plane in board/world frame
    R, _ = cv2.Rodrigues(rvec)
    Rt = R.T
    b = Rt @ tvec  # R^T * t

    # Convert pixel -> normalized bearing using fisheye.undistortPoints (no P)
    pts = np.array(clicks, dtype=np.float64).reshape(-1, 1, 2)
    rays = cv2.fisheye.undistortPoints(pts, K, D)  # shape (N,1,2) normalized
    rays = rays.reshape(-1, 2)

    results = []
    for (u, v), (x, y) in zip(clicks, rays):
        v_cam = np.array([x, y, 1.0], dtype=np.float64).reshape(3, 1)
        A = Rt @ v_cam  # direction expressed in world (board) frame
        Az = float(A[2, 0])
        bz = float(b[2, 0])
        if abs(Az) < 1e-12:
            # Ray parallel to plane -> skip
            world = (np.nan, np.nan, 0.0)
            print("skip")
        else:
            d = bz / Az
            Pw = d * A - b  # world point, Z should be 0
            world = (float(Pw[0, 0]), float(Pw[1, 0]), float(Pw[2, 0]))

        # Annotate world coordinates next to the pixel
        label = f"W=({world[0]:.1f},{world[1]:.1f},{world[2]:.1f})"
        px, py = int(u), int(v)
        cv2.putText(annotated_img_base, label, (px + 6, py + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2, cv2.LINE_AA)
        results.append({
            'pixel': (int(u), int(v)),
            'world': world,
        })

    # Show final annotated image with world labels
    scaled_cache = recompute_scaled_cache()
    refresh_display()

    out_path = './eval_extrinsic_annotated.jpg'
    cv2.imwrite(out_path, annotated_img_base)
    print(f"[eval_extrinsic] Saved annotated image: {out_path}")
    cv2.waitKey(0)
    cv2.destroyWindow(window)
    for ret in results:
        print(f"{ret['pixel']} =>\t {ret['world']}")
    return results

def extrinsic_calibration(img, K, D, chessboardsize, square_size):
    # Given intrinsics (K, D for fisheye), estimate extrinsic pose from a
    # single chessboard image and draw the board axes for visualization.
    # Returns a dict with pose, reprojection error, and an annotated image.
    if img is None or img.size == 0:
        raise ValueError("Invalid input image")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1) Detect corners (fast flags + sub-pixel refinement)
    flags = (
        cv2.CALIB_CB_ADAPTIVE_THRESH
        | cv2.CALIB_CB_FAST_CHECK
        | cv2.CALIB_CB_NORMALIZE_IMAGE
    )
    found, corners = cv2.findChessboardCorners(gray, chessboardsize, flags)
    if not found:
        raise RuntimeError("Chessboard corners not found")

    cv2.cornerSubPix(
        gray,
        corners,
        (3, 3),
        (-1, -1),
        (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.1),
    )
    
    # 2) Build object points in the chessboard frame (Z=0 plane)
    cols, rows = chessboardsize
    objp = np.zeros((cols * rows, 3), np.float32)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2)
    objp *= float(square_size)

    # 3) Undistort corner measurements to an equivalent pinhole camera with K
    #    This avoids full image remap and lets us use solvePnP without distortion.
    undist_corners = cv2.fisheye.undistortPoints(corners, K, D, P=K)

    # 4) Solve PnP for pose (object -> camera): [u v] ~ K [R|t] X
    ok, rvec, tvec = cv2.solvePnP(
        objp,
        undist_corners,
        K,
        None,
        #flags=cv2.SOLVEPNP_ITERATIVE,
        flags=cv2.SOLVEPNP_IPPE
    )
    if not ok:
        raise RuntimeError("solvePnP failed")

    # Draw corners and chessboard axes, then display
    vis = img.copy()
    cv2.drawChessboardCorners(vis, chessboardsize, corners, True)
    axis_len = float(square_size) * 3.0
    axis_obj = np.array(
        [
            [0.0, 0.0, 0.0],
            [axis_len, 0.0, 0.0],   # X - red
            [0.0, axis_len, 0.0],   # Y - green
            [0.0, 0.0, -axis_len],  # Z - blue
        ],
        dtype=np.float32,
    )
    axis_img, _ = cv2.fisheye.projectPoints(axis_obj.reshape(1, -1, 3), rvec, tvec, K, D)
    axis_img = axis_img.reshape(-1, 2).astype(int)
    o, x, y, z = axis_img
    cv2.line(vis, tuple(o), tuple(x), (0, 0, 255), 2)
    cv2.line(vis, tuple(o), tuple(y), (0, 255, 0), 2)
    cv2.line(vis, tuple(o), tuple(z), (255, 0, 0), 2)

    # Draw axis labels at endpoints: X (red), Y (green), Z (blue)
    ox, oy = int(o[0]), int(o[1])
    def put_axis_label(pt, text, color):
        px, py = int(pt[0]), int(pt[1])
        dx, dy = px - ox, py - oy
        offx = 6 if dx >= 0 else -12
        offy = -6 if dy <= 0 else 12
        cv2.putText(
            vis,
            text,
            (px + offx, py + offy),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            color,
            2,
            cv2.LINE_AA,
        )

    put_axis_label(x, 'X', (0, 0, 255))
    put_axis_label(y, 'Y', (0, 255, 0))
    put_axis_label(z, 'Z', (255, 0, 0))
    cv2.imshow('extrinsic_view', vis)
    cv2.imwrite('./ext_corner.jpg', vis)
    cv2.waitKey(0)

    # 5) Compute mean reprojection error on undistorted image points
    proj, _ = cv2.projectPoints(objp, rvec, tvec, K, None)
    err = np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - undist_corners.reshape(-1, 2)) ** 2, axis=1)))

    # 6) Compute roll/pitch/yaw from rotation matrix (ZYX order)
    R, _ = cv2.Rodrigues(rvec)
    sy = np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2)
    singular = sy < 1e-6
    if not singular:
        roll = np.degrees(np.arctan2(R[2, 1], R[2, 2]))  # x-axis
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))     # y-axis
        yaw = np.degrees(np.arctan2(R[1, 0], R[0, 0]))   # z-axis
    else:
        roll = np.degrees(np.arctan2(-R[1, 2], R[1, 1]))
        pitch = np.degrees(np.arctan2(-R[2, 0], sy))
        yaw = 0.0

    # 7) Visualization already performed above; keep annotated image for output

    result = {
        "rvec": rvec,
        "tvec": tvec,
        "txyz": (float(tvec[0]), float(tvec[1]), float(tvec[2])),
        "rpy": (float(roll), float(pitch), float(yaw)),
        "reproj_error": float(err),
        "annotated": vis,
    }
    print(f"{result}")
    return result


if __name__ == '__main__':
    # Calibrate on data5 (paths relative to this file's directory)
    base_dir = os.path.dirname(__file__)
    data_dir = os.path.join(base_dir, 'data7')
    DIM, K, D = get_K_and_D((8,6), data_dir)

    # Undistort every image under data5 into ./undist (beside this script)
    undist_dir = os.path.join(base_dir, 'undist')
    os.makedirs(undist_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(data_dir, '*.jpg')))
    print(f"Undistorting {len(images)} images to: {undist_dir}")
    for img_path in images:
        try:
            out = undistort(img_path, K, D, DIM)
            out_path = os.path.join(undist_dir, os.path.basename(img_path))
            cv2.imwrite(out_path, out)
            print(f"[undist] wrote: {out_path}")
        except Exception as e:
            print(f"[undist][skip] {img_path} due to error: {e}")

    print("Eval project error:")
    test_img_dir = "./data6"
    get_undistor_reproject_err(test_img_dir, K, D, (8, 6), 23, k_scale=0.5)


    print("Extrinsic para calibration...")
    ext_img_path = "/home/zyb/avm/py-OCamCalib/test_images/ext_test/ext_test2.jpg"
    img = cv2.imread(ext_img_path)
    result, rvec, tvec = extrinsic_calibration2(img, K, D, (6,4), 200.0)
    #
    # extrinsic_calibration_fisheye_fixed_kd(ext_img_path, K, D, (6,4), 200.0,
    #                                       prior_tvec=[0, 0, 1200], weight_t=[0, 0, 1.0])

    print("Eval Ext.")
    eval_extrinsic(img, K, D, rvec, tvec)
