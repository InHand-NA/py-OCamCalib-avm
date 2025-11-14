"""Capture 4 cameras simultaneously and display as a 2x2 grid.

Click the on-screen "拍照" button (or press key 'c') to grab
one frame from each camera in order and save images.

The program is a thin Typer CLI wrapper; flags let you choose device
nodes/indices, resolution, FPS, and output directory.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import cv2 as cv
import numpy as np
import typer
from loguru import logger


def _open_capture(dev: str | int, width: int, height: int, fps: int) -> cv.VideoCapture:
    cap = cv.VideoCapture(dev, cv.CAP_V4L2) if isinstance(dev, str) else cv.VideoCapture(int(dev))
    cap.set(cv.CAP_PROP_FOURCC, cv.VideoWriter_fourcc(*'MJPG'))  # 关键：压缩省带宽
    if not cap.isOpened():
        return cap
    if width > 0:
        cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    if height > 0:
        cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    if fps > 0:
        cap.set(cv.CAP_PROP_FPS, fps)
    
    #cap.set(cv.CAP_PROP_BUFFERSIZE, 2)  # 小缓冲，减少阻塞
    return cap


def _as_device(s: str) -> str | int:
    s2 = s.strip()
    if s2.isdigit():
        return int(s2)
    return s2


def _label_tile(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv.putText(out, text, (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
    cv.putText(out, text, (10, 28), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 1, cv.LINE_AA)
    return out


def _make_grid(a: np.ndarray, b: np.ndarray, c: np.ndarray, d: np.ndarray) -> np.ndarray:
    h, w = a.shape[:2]
    grid = np.zeros((2 * h, 2 * w, 3), dtype=np.uint8)
    grid[0:h, 0:w] = a
    grid[0:h, w:2 * w] = b
    grid[h:2 * h, 0:w] = c
    grid[h:2 * h, w:2 * w] = d
    return grid


def _ensure_size(img: Optional[np.ndarray], w: int, h: int) -> np.ndarray:
    if img is None or img.size == 0:
        return np.zeros((h, w, 3), dtype=np.uint8)
    if img.shape[1] != w or img.shape[0] != h:
        return cv.resize(img, (w, h))
    return img


def _save_group(frames: Dict[str, np.ndarray], out_dir: Path) -> List[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    saved: List[Path] = []
    order = ["front", "right", "back", "left"]
    for name in order:
        if name not in frames or frames[name] is None:
            logger.warning(f"No frame for {name}; skipping save")
            continue
        path = out_dir / f"{name}_{ts}.jpg"
        cv.imwrite(str(path), frames[name])
        saved.append(path)
    return saved


def _notify_save(win_name: str, out_dir: Path, ok: bool) -> None:
    """Notify user with a popup/overlay about the save directory."""
    msg = f"Images are saved to: {out_dir}" if ok else f"Image saving fail: {out_dir}"
    # Prefer OpenCV overlay if available
    if hasattr(cv, "displayOverlay"):
        try:
            cv.displayOverlay(win_name, msg, 3000)  # 3s
            return
        except Exception:
            pass
    # Fallback: small popup window
    w = max(360, min(1000, 20 * len(str(out_dir))))
    canvas = np.zeros((80, int(w), 3), dtype=np.uint8)
    cv.putText(canvas, msg, (10, 50), cv.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv.LINE_AA)
    name = "Saving"
    cv.namedWindow(name, cv.WINDOW_AUTOSIZE)
    cv.imshow(name, canvas)
    cv.waitKey(1500)
    cv.destroyWindow(name)

def main(
    front: str = typer.Option("/dev/video10", help="Front camera device (path or index)"),
    right: str = typer.Option("/dev/video12", help="Right camera device (path or index)"),
    back: str = typer.Option("/dev/video14", help="Back camera device (path or index)"),
    left: str = typer.Option("/dev/video16", help="Left camera device (path or index)"),
    width: int = typer.Option(1280, help="Per-camera capture width"),
    height: int = typer.Option(720, help="Per-camera capture height"),
    fps: int = typer.Option(30, help="Per-camera FPS"),
    output_dir: Path = typer.Option(Path("./outputs/captures"), help="Directory to save captured images"),
):
    """Open 4 cameras, show a live 2x2 grid, and capture 4 frames on click."""
    devices: Dict[str, str | int] = {
        "front": _as_device(front),
        "right": _as_device(right),
        "back": _as_device(back),
        "left": _as_device(left),
    }

    caps: Dict[str, cv.VideoCapture] = {name: _open_capture(dev, width, height, fps) for name, dev in devices.items()}
    for name, cap in caps.items():
        if not cap or not cap.isOpened():
            logger.warning(f"Camera '{name}' failed to open: {devices[name]}")

    # Probe one frame to fix tile size
    tile_w, tile_h = width, height
    for name, cap in caps.items():
        if cap and cap.isOpened():
            ok, frm = cap.read()
            if ok and frm is not None and frm.size > 0:
                tile_h, tile_w = frm.shape[:2]
                break

    # 生成一张空白图像，用于缺省填充 frames
    blank_tile = np.zeros((tile_h, tile_w, 3), dtype=np.uint8)

    win = "AVM 2x2 Preview"
    cv.namedWindow(win, cv.WINDOW_NORMAL | cv.WINDOW_KEEPRATIO)

    # Define clickable button rect (drawn on the mosaic).
    # Will be placed at bottom center.
    button_rect: Tuple[int, int, int, int] = (0, 0, 0, 0)
    capture_request = {"flag": False}

    def _on_mouse(event, x, y, flags, param):
        if event == cv.EVENT_LBUTTONDOWN:
            x0, y0, x1, y1 = button_rect
            if x0 < x < x1 and y0 < y < y1:
                capture_request["flag"] = True

    cv.setMouseCallback(win, _on_mouse)

    try:
        log_err_first = dict()
        while True:
            # 缺省用同一张空白图像填充 4 个视图
            frames: Dict[str, np.ndarray] = {
                "front": blank_tile.copy(),
                "right": blank_tile.copy(),
                "back": blank_tile.copy(),
                "left": blank_tile.copy(),
            }
            for name, cap in caps.items():
                ok, frm = (False, None)
                #if name == "right":
                #    continue
                if cap and cap.isOpened():
                    ok, frm = cap.read()
                    if not ok and  not log_err_first.get(name, False):
                        print(f"capture {name} error")
                        log_err_first[name] = True 
                frames[name] = _ensure_size(frm if ok else None, tile_w, tile_h)
                frames[name] = _label_tile(frames[name], name)

            grid = _make_grid(frames["front"], frames["right"], frames["back"], frames["left"])

            # Draw capture button on grid
            gh, gw = grid.shape[:2]
            btn_w, btn_h = 160, 50
            x0 = gw // 2 - btn_w // 2
            y0 = gh - btn_h - 20
            x1 = x0 + btn_w
            y1 = y0 + btn_h
            button_rect = (x0, y0, x1, y1)
            cv.rectangle(grid, (x0, y0), (x1, y1), (0, 180, 255), thickness=2)
            cv.putText(grid, "Picture", (x0 + 40, y0 + 33), cv.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 3, cv.LINE_AA)
            cv.putText(grid, "Picture", (x0 + 40, y0 + 33), cv.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv.LINE_AA)
            cv.putText(grid, "[C]", (x1 + 10 if x1 + 50 < gw else x0 - 50, y0 + 33),
                       cv.FONT_HERSHEY_SIMPLEX, 0.7, (230, 230, 230), 2, cv.LINE_AA)

            cv.imshow(win, grid)

            key = cv.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key in (ord('c'), ord('C')):
                capture_request["flag"] = True

            if capture_request["flag"]:
                # Capture one fresh frame from each camera in order and save
                order = ["front", "right", "back", "left"]
                group_dir = output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
                fresh: Dict[str, np.ndarray] = {}
                for name in order:
                    cap = caps.get(name)
                    frm = None
                    if cap and cap.isOpened():
                        ok, f = cap.read()
                        if ok and f is not None:
                            frm = f
                    if frm is None:
                        logger.warning(f"Capture failed for {name}")
                        frm = frames[name]  # fallback to last displayed frame (resized)
                    fresh[name] = frm

                paths = _save_group(fresh, group_dir)
                if paths:
                    logger.info("Saved: \n" + "\n".join(str(p) for p in paths))
                _notify_save(win, group_dir, ok=bool(paths))
                capture_request["flag"] = False

    finally:
        for cap in caps.values():
            try:
                cap.release()
            except Exception:
                pass
        cv.destroyAllWindows()


if __name__ == "__main__":
    typer.run(main)
