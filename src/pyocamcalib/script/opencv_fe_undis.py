import numpy as np
import cv2
import os
import vvdutils as vv

class FisheyeCameraModel(object):
    """
    Fisheye camera model, for undistorting, projecting and flipping camera frames.
    """

    def __init__(self, camera_param_file, camera_name):
        if not os.path.isfile(camera_param_file):
            raise ValueError("Cannot find camera param file")

        # if camera_name not in settings.camera_names:
        #     raise ValueError("Unknown camera name: {}".format(camera_name))

        self.camera_file = camera_param_file
        self.camera_name = camera_name
        self.scale_xy = (1.0, 1.0)
        self.shift_xy = (0, 0)
        self.undistort_maps = None
        self.project_matrix = None
        #self.project_shape = (3200, 3200)  # settings.project_shapes[self.camera_name]
        self.project_shape = (1080, 1920)  # settings.project_shapes[self.camera_name]
        self.load_camera_params()

    def load_camera_params(self):
        fs = cv2.FileStorage(self.camera_file, cv2.FILE_STORAGE_READ)
        self.camera_matrix = fs.getNode("camera_matrix").mat()
        self.dist_coeffs = fs.getNode("dist_coeffs").mat()
        self.resolution = fs.getNode("resolution").mat().flatten()
        self.camera_type = fs.getNode("camera_type").string()

        if fs.getNode("scale_xy").mat() is not None:
            self.scale_xy = fs.getNode("scale_xy").mat().flatten()

        if fs.getNode("shift_xy").mat() is not None:
            self.shift_xy = fs.getNode("shift_xy").mat().flatten()

        if fs.getNode("project_matrix").mat() is not None:
            self.project_matrix = fs.getNode("project_matrix").mat()

        fs.release()
        self.update_undistort_maps()

    def update_undistort_maps(self):
        new_matrix = self.camera_matrix.copy()
        new_matrix[0, 0] *= self.scale_xy[0]
        new_matrix[1, 1] *= self.scale_xy[1]
        new_matrix[0, 2] += self.shift_xy[0]
        new_matrix[1, 2] += self.shift_xy[1]
        width, height = self.resolution

        self.undistort_maps = cv2.fisheye.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            np.eye(3),
            new_matrix,
            (width, height),
            cv2.CV_16SC2
        )
        return self

    def set_scale_and_shift(self, scale_xy=(1.0, 1.0), shift_xy=(0, 0)):
        self.scale_xy = scale_xy
        self.shift_xy = shift_xy
        self.update_undistort_maps()
        return self

    def undistort(self, image):
        result = cv2.remap(image, *self.undistort_maps, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        return result

    def project(self, image):
        if self.project_matrix is None:
            return image.copy()
        result = cv2.warpPerspective(image, self.project_matrix, self.project_shape)
        return result

    def flip(self, image):
        if self.camera_name == "front":
            return image.copy()

        elif self.camera_name == "back":
            return image.copy()[::-1, ::-1, :]

        elif self.camera_name == "left":
            return cv2.transpose(image)[::-1]

        else:
            return np.flip(cv2.transpose(image), 1)

    def save_data(self):
        fs = cv2.FileStorage(self.camera_file, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", self.camera_matrix)
        fs.write("dist_coeffs", self.dist_coeffs)
        fs.write("resolution", self.resolution)
        fs.write("project_matrix", self.project_matrix)
        fs.write("scale_xy", np.float32(self.scale_xy))
        fs.write("shift_xy", np.float32(self.shift_xy))
        fs.release()


 

class NormalCameraModel(object):
    """
    normal camera model, for undistorting, projecting 
    """

    def __init__(self, camera_param_file, camera_name):
        if not os.path.isfile(camera_param_file):
            raise ValueError("找不到相机参数文件")

        self.camera_file = camera_param_file
        self.camera_name = camera_name
        self.undistort_map = None
        self.scale_xy = (1.0, 1.0)
        self.shift_xy = (0, 0)
        self.load_camera_params()

    def load_camera_params(self):
        fs = cv2.FileStorage(self.camera_file, cv2.FILE_STORAGE_READ)
        self.camera_matrix = fs.getNode("camera_matrix").mat()
        self.dist_coeffs = fs.getNode("dist_coeffs").mat()
        self.resolution = fs.getNode("resolution").mat().flatten()
        if fs.getNode("scale_xy").mat() is not None:
            self.scale_xy = fs.getNode("scale_xy").mat().flatten()

        if fs.getNode("shift_xy").mat() is not None:
            self.shift_xy = fs.getNode("shift_xy").mat().flatten()

        if fs.getNode("project_matrix").mat() is not None:
            self.project_matrix = fs.getNode("project_matrix").mat()

        fs.release()
        self.update_undistort_map()
    
    def set_scale_and_shift(self, scale_xy=(1.0, 1.0), shift_xy=(0, 0)):
        self.scale_xy = scale_xy
        self.shift_xy = shift_xy
        self.update_undistort_map()
        return self

    def update_undistort_map(self):
        new_matrix = self.camera_matrix.copy()
        new_matrix[0, 0] *= self.scale_xy[0]
        new_matrix[1, 1] *= self.scale_xy[1]
        new_matrix[0, 2] += self.shift_xy[0]
        new_matrix[1, 2] += self.shift_xy[1]
        width, height = self.resolution

        width, height = self.resolution
        self.undistort_map = cv2.initUndistortRectifyMap(
            self.camera_matrix,
            self.dist_coeffs,
            np.eye(3, 3),
            new_matrix,
            (width, height),
            cv2.CV_16SC2
        )

    def undistort(self, image):
        result = cv2.remap(image, *self.undistort_map, interpolation=cv2.INTER_LINEAR,
                           borderMode=cv2.BORDER_CONSTANT)
        return result

    def save_data(self):
        fs = cv2.FileStorage(self.camera_file, cv2.FILE_STORAGE_WRITE)
        fs.write("camera_matrix", self.camera_matrix)
        fs.write("dist_coeffs", self.dist_coeffs)
        fs.write("resolution", self.resolution)
        fs.release()


# 示例用法
camera_model = FisheyeCameraModel("camera_intrinsic.yaml", "camera_0")
# 设置新的缩放和偏移
camera_model.set_scale_and_shift(scale_xy=(1.0,1.0), shift_xy=(0, 0))
camera_model.update_undistort_maps()

image_path_list = vv.glob_images('./test_images/inhandus_1/')
for image_path in image_path_list:    
    distorted_image= cv2.imread(image_path)
    undistorted_image = camera_model.undistort(distorted_image)
    print(f"file: {image_path}")
    cv2.imshow("undistorted", undistorted_image)
    cv2.waitKey(0)
    save_path = vv.OS_join('result', vv.OS_basename(image_path))
    vv.cv_bgr_imwrite(undistorted_image, save_path)
    vv.PIS(undistorted_image)

# camera_model.save_data()
pass
