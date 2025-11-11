"""
     Hugo Vazquez email: hugo.vazquez@jakarto.com
     Copyright (C) 2022  Hugo Vazquez

     This program is free software; you can redistribute it and/or modify
     it under the terms of the GNU General Public License as published by
     the Free Software Foundation; either version 2 of the License, or
     (at your option) any later version.

     This program is distributed in the hope that it will be useful,
     but WITHOUT ANY WARRANTY; without even the implied warranty of
     MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
     GNU General Public License for more details.

     You should have received a copy of the GNU General Public License along
     with this program; if not, write to the Free Software Foundation, Inc.,
     51 Franklin Street, Fifth Floor, Boston, MA 02110-1301 USA.
"""

from pathlib import Path
from typing import List, Tuple

import typer
import cv2 as cv
import matplotlib.pyplot as plt

from pyocamcalib.modelling.camera import Camera

OUTPUT_DIR = Path("./outputs")


def _gather_image_paths(fisheye_path: Path) -> List[Path]:
    if fisheye_path.is_dir():
        images = sorted(
            p for p in fisheye_path.iterdir()
            if p.is_file() and p.suffix.lower() == ".jpg"
        )
        if not images:
            raise typer.BadParameter(
                f"No .jpg files were found in directory {fisheye_path}"
            )
        return images
    if fisheye_path.is_file():
        return [fisheye_path]
    raise typer.BadParameter(f"{fisheye_path} is not a valid file or directory.")


def _save_projection(image_path: Path,
                     camera: Camera,
                     perspective_fov: float,
                     perspective_sensor_size: Tuple[int, int],
                     show_plot: bool) -> Path:
    fisheye_image = cv.imread(str(image_path))
    if fisheye_image is None:
        raise typer.BadParameter(f"Unable to read image: {image_path}")

    perspective_image = camera.cam2perspective_indirect(
        fisheye_image,
        perspective_fov,
        perspective_sensor_size,
    )

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_DIR / f"{image_path.stem}_proj.jpg"
    cv.imwrite(str(output_path), perspective_image)

    if show_plot:
        plt.figure()
        plt.imshow(fisheye_image[:, :, ::-1])
        plt.title("Original fisheye image")
        plt.figure()
        plt.imshow(perspective_image[:, :, ::-1])
        plt.title(f"Perspective conversion. fov = {perspective_fov} deg")
        plt.show()

    return output_path


def main(fisheye_image_path: str,
         calibration_file_path: str,
         perspective_fov: float,
         perspective_sensor_size: Tuple[int, int],
         ):
    """

    :param fisheye_image_path: fisheye image path or directory containing .jpg files.
    :param calibration_file_path: .json file with calibration parameters.
    :param perspective_fov: field of view the desired perspective camera in degree (between 0 and 180).
    :param perspective_sensor_size: (height, width) in pixels. Determine the output image resolution.
    :return:
    """

    target_path = Path(fisheye_image_path)
    image_paths = _gather_image_paths(target_path)
    camera = Camera.load_parameters_json(calibration_file_path)

    for image_path in image_paths:
        output_path = _save_projection(
            image_path=image_path,
            camera=camera,
            perspective_fov=perspective_fov,
            perspective_sensor_size=perspective_sensor_size,
            show_plot=len(image_paths) == 1,
        )
        print(f"Image is saved at: {output_path}")


if __name__ == "__main__":
    typer.run(main)
